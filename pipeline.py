import os
import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
import torch.nn.functional as F
from basic_module import Transformer
import utils
from pretrained_models.dpv2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt  

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("Warning: 'segment_anything' not installed. SAM features will be disabled.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleGallery(StableDiffusionPipeline):
    def freeze(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.extractor.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = self.vae.encode(image)["latent_dist"].mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def init(self, enable_gradient_checkpoint):
        self.freeze()
        self.enable_vae_slicing()
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.extractor.to(self.accelerator.device, dtype=weight_dtype)
        # self.extractor = self.accelerator.prepare(self.extractor)
        if enable_gradient_checkpoint:
            self.extractor.enable_gradient_checkpointing()

    def generate_mask_with_sam(self, image, points_per_side=16, pred_iou_thresh=0.98):
        if not HAS_SAM:
            raise ImportError("Please install segment-anything to use SAM features.")
        
        sam_checkpoint_h = "segment-anything/ckpts/sam_vit_h_4b8939.pth"
        sam_checkpoint_b = "segment-anything/ckpts/sam_vit_b_01ec64.pth"
        
        if os.path.exists(sam_checkpoint_h):
            sam_checkpoint = sam_checkpoint_h
            model_type = "vit_h"
        elif os.path.exists(sam_checkpoint_b):
            sam_checkpoint = sam_checkpoint_b
            model_type = "vit_b"
        else:
            raise FileNotFoundError(f"SAM checkpoint not found.")

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000, 
        )
        
        if image is None:
            raise ValueError(f"Could not load image")

        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().squeeze().numpy()
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.max() <= 1.0:
                img_np = img_np * 255.0
            sam_image = np.clip(img_np, 0, 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            sam_image = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            sam_image = image
            if sam_image.ndim == 3 and sam_image.shape[0] == 3:
                sam_image = np.transpose(sam_image, (1, 2, 0))
            if sam_image.max() <= 1.0:
                sam_image = sam_image * 255.0
            sam_image = np.clip(sam_image, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type")

        masks = mask_generator.generate(sam_image)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        H, W, _ = sam_image.shape
        mask_map = np.zeros((H, W), dtype=np.int32)

        for i, ann in enumerate(sorted_anns):
            m = ann['segmentation']
            mask_map[m] = i + 1

        mask_tensor = torch.from_numpy(mask_map).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_64 = F.interpolate(mask_tensor, size=(64, 64), mode='nearest')
        mask_64_np = mask_64.squeeze().cpu().numpy().astype(np.int32)
        
        del sam
        del mask_generator
        torch.cuda.empty_cache()
        
        num_clusters = len(sorted_anns)
        return mask_64_np, num_clusters

    @torch.no_grad()
    def feature_cluster(
        self,
        features,
        latent_feature,
        depth_feature,
        max_clusters=10,
        random_seed=42,
        original_mask=None,
        sam_mask=None
    ):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        features_2d = features.squeeze(0).permute(1, 2, 0).reshape(-1, features.shape[1]).to(device)
        latent_feature = latent_feature.squeeze(0).permute(1, 2, 0).reshape(-1, 4).to(device)

        if original_mask is not None:
            if isinstance(original_mask, np.ndarray):
                original_mask = torch.from_numpy(original_mask).to(device)
            elif original_mask.device != device:
                original_mask = original_mask.to(device)

        with torch.no_grad():
            mean = features_2d.mean(dim=0)
            std = features_2d.std(dim=0)
            X = (features_2d - mean) / std  # [N, C]

            n_components = min(features.shape[1], 60)
            U, S, V = torch.pca_lowrank(X, q=n_components, center=False)
            X_pca = X @ V[:, :n_components]

            if sam_mask is not None:
                if isinstance(sam_mask, np.ndarray):
                    sam_mask = torch.from_numpy(sam_mask).to(device)
                flat_sam = sam_mask.view(-1).long()
                unique_labels, cluster_labels = torch.unique(flat_sam, return_inverse=True)
                num_clusters = unique_labels.numel()
                centroids = torch.zeros((num_clusters, X_pca.shape[1]), device=device)
                for i in range(num_clusters):
                    m = (cluster_labels == i)
                    if m.any():
                        centroids[i] = X_pca[m].mean(dim=0)
                new_cluster_labels, new_centroids = self.merge_similar_clusters(
                    cluster_labels, centroids, latent_feature, X_pca, similar_threshold=0.5
                )
            else:
                silhouette_scores = []
                labels_per_k = []
                for k in tqdm(range(2, int(max_clusters + 1))):
                    labels_k, cents_k = self.gpu_kmeans(X_pca, k, n_init=10, seed=random_seed)
                    labels_per_k.append((labels_k, cents_k))
                    score = self.silhouette_score_gpu(X_pca, labels_k, sample_size=5000)
                    silhouette_scores.append(score)

                optimal_clusters = int(np.argmax(silhouette_scores)) + 2
                cluster_labels, centroids = labels_per_k[optimal_clusters - 2]

                new_cluster_labels, new_centroids = self.merge_similar_clusters(
                    cluster_labels, centroids, latent_feature, X_pca, similar_threshold=0.85
                )

                if depth_feature is not None:
                    depth_cluster_labels, depth_new_centroids = self.refine_with_depth(
                        new_cluster_labels, new_centroids, depth_feature, X_pca, split_threshold=0.75, seed=random_seed
                    )
                    new_cluster_labels, new_centroids = self.merge_similar_clusters(
                        depth_cluster_labels, depth_new_centroids, latent_feature, X_pca, similar_threshold=0.35, depth_feature=depth_feature, depth_threshold=0.3
                    )

            cluster_mask = new_cluster_labels.detach().cpu().numpy().reshape(64, 64) + 1
            final_clusters = int(torch.unique(new_cluster_labels).numel())

            if original_mask is not None:
                cluster_mask = cluster_mask - 1
                cluster_mask, new_clusters = self.update_clusters_with_mask(torch.from_numpy(cluster_mask).to(device), original_mask)
                final_clusters = int(torch.unique(cluster_mask).numel())
                cluster_mask = cluster_mask + 1

            return cluster_mask, final_clusters

    def generate_mask(self, unet_latent, image, latent, depth=None, image_name=None, original_mask=None, use_sam=False):
        sam_base_mask = None
        if use_sam and HAS_SAM:
            print(f"[{image_name}] Using Segment Anything (SAM) for base mask generation...")
            sam_base_mask, _ = self.generate_mask_with_sam(image)
            
        mask, _ = self.feature_cluster(
            unet_latent, 
            latent, 
            depth, 
            original_mask=original_mask,
            sam_mask=sam_base_mask
        )
        refine_mask = utils.remove_small_regions(mask, self.device, min_size=20)
        utils.visualize_clustering(
            unet_latent,
            refine_mask,
            image,
            image_name=image_name
        )
        return refine_mask

    def weighted_features(self, latent, total_elements=15):
        def adaptive_decay(x, total_elements):
            return 1 / (1 + torch.exp(5 * (x / total_elements - 0.7)))
        
        indices = torch.arange(total_elements, device=latent[0].device)
        weights = adaptive_decay(indices, total_elements)
        weights = weights / weights.sum()
        weighted_feature = torch.stack(latent).mul(weights[:, None, None, None, None]).sum(dim=0)

        return weighted_feature

    def forward_process(self, latent, text_embeds, steps=15):
        pred_images = []
        pred_latents = []
        unet_feature = []

        self.scheduler.set_timesteps(steps)
        timesteps = reversed(self.scheduler.timesteps)
        cur_latent = latent.clone()
        with torch.no_grad():
            for i in tqdm(range(0, steps), desc="DDIM Inversion"):
                t = timesteps[i]
                Unetcache = utils.UnetDataCache()
                hooks = utils.register_unet_feature_extraction(
                    self.unet, Unetcache
                )
                current_t = max(0, t.item() - (1000 // steps))
                next_t = t.item()
                next_t_tensor = torch.tensor([next_t], dtype=torch.long, device=device)
                noise_pred = self.unet(
                    cur_latent,
                    next_t_tensor, 
                    text_embeds
                ).sample
                _, unet_feature_2 = Unetcache.get_features()
                unet_feature.append(unet_feature_2)

                for hook in hooks:
                    hook.remove()

                alpha_t = self.scheduler.alphas_cumprod[torch.tensor(current_t, dtype=torch.long)]
                alpha_t_next = self.scheduler.alphas_cumprod[torch.as_tensor(next_t, dtype=torch.long).clone().detach()]
                
                cur_latent = (cur_latent - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred
                pred_latents.append(cur_latent)
                if i % 3 == 0:
                    pred_images.append(self.latent2image(cur_latent))

        unet_up_block_feature2 = self.weighted_features(unet_feature, steps)
        return pred_images, pred_latents, unet_up_block_feature2

    def extract_semantic_features(self, image_path, mask, dinov2_model, processor):
        mask = mask.float()
        enlarged_mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode="nearest"
        )
        # [1, 1, 512, 512] -> [1, 256, 32, 32] -> [1, 256, 1024]
        blocked_mask = enlarged_mask.unfold(2, 32, 32).unfold(3, 32, 32)
        blocked_mask = blocked_mask.contiguous().view(1, -1, 32, 32)
        blocked_mask = blocked_mask.view(1, 256, -1)

        valid_counts = blocked_mask.view(1, 256, 1024).sum(dim=2)
        keep_indices = (valid_counts > 10).squeeze()
        new_mask = torch.zeros(256, dtype=torch.int)
        new_mask[keep_indices] = 1
        image = Image.open(image_path).convert('RGB')

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
            # [1, 256, 768]
            tokens = outputs.last_hidden_state[:, 1:, :]
        preserved_tokens = tokens[:, new_mask.bool(), :]
        return preserved_tokens.to(device)

    def match_content_style_clusters(
        self,
        content_features,
        content_mask,
        content_image_path,
        style_feature_mask_dict
    ):
        transformer = Transformer(
            width=content_features.shape[1],
            layers=8,
            heads=4,
        ).to(content_features.device)

        dinov2_model = AutoModel.from_pretrained("pretrained_models/facebook/dinov2-base").to(self.device)
        dinov2_processor = AutoImageProcessor.from_pretrained("pretrained_models/facebook/dinov2-base")

        def extract_semantic_features(features, mask, image_path):
            cluster_data = {}
            mask = mask.to(self.device)
            unique_clusters = torch.unique(mask)

            for cluster in unique_clusters:
                c_mask = (mask == cluster)
                valid_indices = torch.nonzero(c_mask.squeeze(), as_tuple=False)
                    
                if valid_indices.size(0) == 0:
                    continue

                masked_features = features * c_mask.unsqueeze(0).float()
                valid_features = masked_features[0, :, valid_indices[:, 0], valid_indices[:, 1]]
                attended_features = transformer(valid_features.t()).t()
                reconstructed_features = torch.zeros_like(masked_features)
                reconstructed_features[0, :, valid_indices[:, 0], valid_indices[:, 1]] = attended_features
                channel_sums = reconstructed_features.sum(dim=(2, 3))
                valid_points = c_mask.sum()
                avg_feature = channel_sums / (valid_points + 1e-8)
                variance_feature = reconstructed_features.var(dim=(2, 3)) / (valid_points + 1e-8)
                combined_feature = torch.cat((avg_feature, variance_feature), dim=1).squeeze()
                points = valid_indices.cpu().numpy()
                center, radius = utils.minimum_enclosing_circle(points)
                dinov2_semantic_feature = self.extract_semantic_features(
                    image_path, c_mask, dinov2_model, dinov2_processor
                ).squeeze()
                cluster_data[cluster.item()] = {
                    'combined_feature': combined_feature,
                    'circle_center': (center[0], center[1]),
                    'circle_radius': radius,
                    'dinov2_feature': dinov2_semantic_feature,
                }
                
            return cluster_data
        
        content_cluster_features = extract_semantic_features(content_features, content_mask, content_image_path)
        style_cluster_features = {}
        for style_dict_index, style_dict in enumerate(style_feature_mask_dict):
            s_data = extract_semantic_features(style_dict['unet_feature'], style_dict['mask'], style_dict['image'])
            for s_cluster, feats in s_data.items():
                style_cluster_features[(style_dict_index, s_cluster)] = feats

        cluster_matches = {}
        for content_cluster, content_data in content_cluster_features.items():
            best_match = {
                'style_dict_index': 0, 'style_cluster': 0, 'similarity': float('-inf')
            }
            c_combined = content_data['combined_feature'].flatten().to(self.device)
            c_dinov2 = content_data['dinov2_feature'].mean(dim=0).to(self.device)
            if c_dinov2.dim() == 0: c_dinov2 = torch.zeros(1, device=self.device)

            for (style_dict_index, style_cluster), style_data in style_cluster_features.items():
                s_combined = style_data['combined_feature'].flatten().to(self.device)
                s_dinov2 = style_data['dinov2_feature'].mean(dim=0).to(self.device)
                if s_dinov2.dim() == 0: s_dinov2 = torch.zeros(1, device=self.device)

                combined_sim = F.cosine_similarity(c_combined.unsqueeze(0), s_combined.unsqueeze(0)).item()
                dinov2_sim = F.cosine_similarity(c_dinov2.unsqueeze(0), s_dinov2.unsqueeze(0)).item()
                overlap_ratio = utils.calculate_overlap_ratio(
                    content_data['circle_center'], content_data['circle_radius'],
                    style_data['circle_center'], style_data['circle_radius']
                )

                similarity = (combined_sim * 0.25) + dinov2_sim + (overlap_ratio * 0.125)
                if similarity > best_match['similarity']:
                    best_match = {
                        'style_dict_index': style_dict_index,
                        'style_cluster': style_cluster,
                        'similarity': similarity
                    }
            if best_match['similarity'] == float('-inf'):
                print(f"警告：未找到内容聚类 {content_cluster} 的有效匹配！")
                best_match['similarity'] = 0.0

            cluster_matches[content_cluster] = best_match

        return cluster_matches

    def cluster_match(self, content_image_path, style_image_path, steps, content_original_mask=None, use_depth=False, use_sam=False):
        null_text_embeds = self.encode_prompt("", self.device, 1, False)[0]
        content_latent = self.image2latent(utils.load_image(content_image_path, (512, 512)))
        text_embeds = null_text_embeds.repeat(content_latent.shape[0], 1, 1)
        content_dict = self.process_image_data(
            image_path=content_image_path,
            latent=content_latent,
            text_embeds=text_embeds,
            steps=steps,
            image_name="content",
            original_mask=content_original_mask,
            use_depth=use_depth,
            use_sam=use_sam
        )
        style_image_paths_list = [style_image_path] if isinstance(style_image_path, str) else style_image_path
        style_dict_list = []
        for i, current_style_path in enumerate(style_image_paths_list):
            current_style_latent = self.image2latent(utils.load_image(current_style_path, (512, 512)))
            
            style_name = os.path.splitext(os.path.basename(current_style_path))[0]
            style_mask_path = os.path.join(os.path.dirname(current_style_path), f"{style_name}_json", "label.png")
            style_original_mask = utils.convert_mask_to_array(style_mask_path) if os.path.exists(style_mask_path) else None

            style_dict = self.process_image_data(
                image_path=current_style_path,
                latent=current_style_latent,
                text_embeds=text_embeds, 
                steps=steps,
                image_name=f"style{i}",
                original_mask=style_original_mask,
                use_depth=use_depth,
                use_sam=use_sam  
            )
            style_dict_list.append(style_dict)

        cluster_matches = self.match_content_style_clusters(
            content_dict['unet_feature'],
            content_dict['mask'],
            content_image_path,
            style_dict_list
        )
        return cluster_matches, content_dict, style_dict_list

    def style_transfer(
        self,
        content_dict,
        style_dict,
        controller,
        cluster_matches,
        mixed_precision="fp16",
        num_optimize_steps=150,
        enable_gradient_checkpoint=False,
        lr=0.05,
        iters=1,
        c_ratio=0.26,
    ):
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self.init(enable_gradient_checkpoint)

        content_latent = content_dict["feature"].to(device).half()
        content_mask   = content_dict["mask"].to(device).half()
        latents = content_dict["C15"].clone().to(device)
        latents = self.remove_style(latents)

        null_text_embeds = self.encode_prompt("", self.device, 1, False)[0]
        text_embeds = null_text_embeds.repeat(content_latent.shape[0], 1, 1).to(device)

        self.attncache = utils.AttnDataCache()
        self.controller = controller
        utils.register_attn_control(
            self.extractor,
            self.controller,
            self.attncache
        )
        self.extractor = self.accelerator.prepare(self.extractor)
        gpu_style_dict = []
        for sd in style_dict:
            gpu_style_dict.append({
                "feature": sd["feature"].to(device),
                "mask":    sd["mask"].to(device)
            })

        self.scheduler.set_timesteps(num_optimize_steps)
        timesteps = self.scheduler.timesteps

        latents = latents.detach().float()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer = self.accelerator.prepare(optimizer)

        unique_content_clusters = torch.unique(content_mask)
        precomputed_c_masks = {
            c.item(): (content_mask == c).float().to(device) for c in unique_content_clusters
        }
        
        precomputed_s_masks = {}
        for c_id, match_info in cluster_matches.items():
            if match_info is None:
                continue
            s_idx = match_info["style_dict_index"]
            s_cluster = match_info["style_cluster"]
            if (s_idx, s_cluster) not in precomputed_s_masks:
                s_mask = gpu_style_dict[s_idx]["mask"]
                precomputed_s_masks[(s_idx, s_cluster)] = (s_mask == s_cluster).float().to(device)

        global_mask_cache = {}
        pbar = tqdm(timesteps, desc="Optimize")

        for i, t in enumerate(pbar):
            with torch.no_grad():
                qc_list, kc_list, vc_list, c_out_list = self.extract_feature(
                    content_latent,
                    t,
                    text_embeds
                )
                style_attn_dict = []
                for sd in gpu_style_dict:
                    style_feat = sd["feature"]
                    style_mask = sd["mask"]
                    q, k, v, s_out_list = self.extract_feature(
                        style_feat,
                        t,
                        text_embeds
                    )
                    style_attn_dict.append({
                        "q": q,
                        "k": k,
                        "v": v,
                        "mask": style_mask,
                        "feature": style_feat
                    })

            for j in range(iters):
                optimizer.zero_grad(set_to_none=True)
                style_loss = 0.0
                q_list, k_list, v_list, self_out_list = self.extract_feature(
                    latents,
                    t,
                    text_embeds
                )
                content_loss = utils.content_loss(q_list, qc_list)
                style_loss = self.get_style_loss(
                       cluster_matches,
                       precomputed_c_masks,
                       precomputed_s_masks,
                       q_list,
                       self_out_list,
                       style_attn_dict,
                       global_mask_cache
                )
                loss = style_loss + c_ratio * content_loss
                self.accelerator.backward(loss)
                optimizer.step()
                pbar.set_postfix(loss=float(loss.detach().item()), time=float(t.item()))

        image = self.latent2image(latents)
        self.maybe_free_model_hooks()
        return image

    def get_style_loss(
        self, 
        cluster_matches, 
        precomputed_c_masks, 
        precomputed_s_masks, 
        self_q,  
        self_out, 
        style_attn_dict, 
        mask_cache
    ):
        style_loss = 0.0
        def apply_mask(mask, tensors):
            results = []
            mask_id = id(mask) 
            for t in tensors:
                size = int((t.shape[2]) ** 0.5)
                key = (mask_id, size)
                
                if key not in mask_cache:
                    scaled_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(size, size), mode="nearest")
                    scaled_mask = scaled_mask.reshape(1, 1, size * size, 1)
                    mask_cache[key] = scaled_mask
                    
                results.append(t * mask_cache[key])
            return results

        for c_id, content_cluster_mask in precomputed_c_masks.items():
            match_info = cluster_matches.get(c_id, None)
            if match_info is None:
                continue
            style_dict_index = match_info["style_dict_index"]
            style_cluster = match_info["style_cluster"]
            style_mask = precomputed_s_masks[(style_dict_index, style_cluster)]
            style_attn = style_attn_dict[style_dict_index]
            style_k = style_attn["k"]
            style_v = style_attn["v"]
            q_self_temp = apply_mask(content_cluster_mask, self_q)
            ks_temp     = apply_mask(style_mask, style_k)
            vs_temp     = apply_mask(style_mask, style_v)
            out_temp    = apply_mask(content_cluster_mask, self_out)
            style_loss += utils.style_loss(q_self_temp, ks_temp, vs_temp, out_temp, scale=1.5)

        return style_loss

    def extract_feature(self, latent, t, embeds):
        self.attncache.clear()
        self.controller.step()
        _ = self.extractor(latent, t, embeds)[0]
        return self.attncache.get()

    def remove_style(self, features):
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        return (features - mean) / (std + 1e-8)

    @torch.no_grad()
    def get_depth(
        self,
        image_path,
        output_dir: str = "./outputs/depth_visual",
        save_colormap=True
    ):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitl' 

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'pretrained_models/dpv2/ckpts/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(self.device).eval()

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1] 
       
        depth = model.infer_image(img)  

        if output_dir:
            # os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, base + "_depth.png")

            d_min, d_max = float(depth.min()), float(depth.max())
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
            d8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
            # vis = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO) if save_colormap else d8
            # cv2.imwrite(save_path, np.ascontiguousarray(vis))
            # print(f"[Saved] depth image: {save_path}")

        return depth
    
    def gpu_kmeans(self, data, k, n_init=10, max_iters=64, tol=1e-4, seed=None):
        g = torch.Generator(device=data.device)
        if seed is not None:
            g.manual_seed(seed)
        N, D = data.shape
        best_inertia, best_labels, best_centroids = torch.inf, None, None
        for _ in range(n_init):
            centroids = data[torch.randperm(N, generator=g, device=data.device)[:k]]
            for _ in range(max_iters):
                x2 = (data**2).sum(1, keepdim=True)  # [N,1]
                c2 = (centroids**2).sum(1)           # [k]
                dist = x2 - 2 * (data @ centroids.T) + c2  # [N,k]
                labels = dist.argmin(1)
                new_centroids = torch.stack([
                    data[labels == i].mean(0) if (labels == i).any()
                    else data[torch.randint(0, N, (1,), device=data.device)].squeeze(0)
                    for i in range(k)
                ])
                shift = (centroids - new_centroids).norm(dim=1).max()
                centroids = new_centroids
                if shift <= tol:
                    break
            inertia = dist.gather(1, labels.view(-1, 1)).sum()
            if inertia < best_inertia:
                best_inertia, best_labels, best_centroids = inertia, labels.clone(), centroids.clone()
        return best_labels, best_centroids

    def silhouette_score_gpu(self, data, labels, sample_size=100000):
        labels = labels.view(-1)
        N = data.size(0)
        if sample_size is not None and N > sample_size:
            idx = torch.randperm(N, device=data.device)[:sample_size]
            data, labels = data[idx], labels[idx]
            N = data.size(0)

        K = int(labels.max().item()) + 1
        D = torch.cdist(data, data)  # [N,N]
        M = F.one_hot(labels, num_classes=K).float()
        cluster_sizes = M.sum(0).clamp_min(1.0)
        S = D @ M
        denom_self = (cluster_sizes[labels] - 1).clamp_min(1.0)
        a = S[torch.arange(N, device=data.device), labels] / denom_self
        avg_to_clusters = S / cluster_sizes
        mask_self = F.one_hot(labels, num_classes=K).bool()
        avg_to_clusters = avg_to_clusters.masked_fill(mask_self, float('inf'))
        b, _ = avg_to_clusters.min(dim=1)

        s = (b - a) / torch.clamp(torch.maximum(a, b), min=1e-12)
        singletons = (cluster_sizes[labels] <= 1)
        s = torch.where(singletons, torch.zeros_like(s), s)
        return float(s.mean().item())
    
    def merge_similar_clusters(self, labels, centroids, features, X_pca, similar_threshold=0.75, depth_feature=None, depth_threshold=None):
        device = labels.device
        if centroids.size(0) == 0:
            return labels, centroids
                
        if labels.max() >= centroids.size(0) or labels.min() < 0:
            labels = labels.clamp(0, centroids.size(0) - 1)
                
        pixel_means = []
        for i in range(centroids.size(0)):
            m = (labels == i)
            if m.any():
                pixel_means.append(features[m].mean(dim=0))
            else:
                if features.size(0) == 0:
                    pixel_means.append(torch.zeros_like(centroids[0]))
                else:
                    pixel_means.append(features[torch.randint(0, features.size(0), (1,), device=device)].squeeze(0))
                
        pixel_means = torch.stack(pixel_means).float()
        pixel_distances = torch.cdist(pixel_means, pixel_means)
        num_clusters = centroids.size(0)
        
        use_depth_lock = (depth_feature is not None) and (depth_threshold is not None)
        if use_depth_lock:
            if isinstance(depth_feature, np.ndarray):
                depth_feature = torch.from_numpy(depth_feature).to(device).float()
            
            depth_64 = F.interpolate(depth_feature.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='nearest').squeeze()
            d_min, d_max = depth_64.min(), depth_64.max()
            depth_flat = (depth_64 - d_min) / (d_max - d_min + 1e-6)
            depth_flat = depth_flat.view(-1)

            depth_means = torch.zeros(num_clusters, device=device)
            for i in range(num_clusters):
                m = (labels == i)
                if m.any():
                    depth_means[i] = depth_flat[m].mean()
            depth_distances = torch.abs(depth_means.unsqueeze(0) - depth_means.unsqueeze(1))

        label_mapping = torch.arange(num_clusters, device=device)
        merge_map = {}
                
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                color_ok = pixel_distances[i, j] < similar_threshold
                depth_ok = (not use_depth_lock) or (depth_distances[i, j] < depth_threshold)
                
                if color_ok and depth_ok:
                    merge_map[j] = i
                
        for remove_label in list(merge_map.keys()):
            keep_label = merge_map[remove_label]
            while keep_label in merge_map:
                keep_label = merge_map[keep_label]
            merge_map[remove_label] = keep_label
            
        for remove_label, keep_label in merge_map.items():
            label_mapping[label_mapping == remove_label] = keep_label
            
        unique_labels, inverse_indices = torch.unique(label_mapping, return_inverse=True)
        label_mapping = inverse_indices
        new_labels = label_mapping[labels]

        new_num_clusters = unique_labels.size(0)
        new_centroids = torch.stack([
            X_pca[new_labels == i].mean(dim=0) if (new_labels == i).any() else torch.zeros(X_pca.shape[1], device=device)
            for i in range(new_num_clusters)
        ])
        
        return new_labels, new_centroids
        
    def refine_with_depth(self, labels, centroids, depth_feature, X_pca, split_threshold=0.4, seed=42):
        if isinstance(depth_feature, np.ndarray):
            depth_feature = torch.from_numpy(depth_feature).to(device).float()
        depth_64 = F.interpolate(depth_feature.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='nearest').squeeze()
                
        d_min, d_max = depth_64.min(), depth_64.max()
        depth_flat = (depth_64 - d_min) / (d_max - d_min + 1e-6)
        depth_flat = depth_flat.view(-1)
                
        unique_clusters = torch.unique(labels)
        new_label_offset = labels.max() + 1
        centroids_list = list(centroids)
                
        for cluster in unique_clusters:
            c_id = cluster.item()
            mask = (labels == c_id)
            indices = torch.nonzero(mask).squeeze(1)
            if indices.numel() < 20: 
                continue
                        
            cluster_depth = depth_flat[indices]
            sorted_depth, _ = torch.sort(cluster_depth)
            p05_idx = int(0.05 * sorted_depth.numel())
            p95_idx = int(0.95 * sorted_depth.numel())
            depth_span = sorted_depth[p95_idx] - sorted_depth[p05_idx]

            if depth_span >= split_threshold:
                print(f"⚠️ [Depth Guardrail] Cluster {c_id} spans extreme depth ({depth_span:.2f}). Splitting...")
                sub_labels, _ = self.gpu_kmeans(cluster_depth.unsqueeze(1), k=2, n_init=1, seed=seed)
                sub0_mask = (sub_labels == 0)
                sub1_mask = (sub_labels == 1)
                if sub0_mask.sum() > 0 and sub1_mask.sum() > 0:
                    labels[indices[sub1_mask]] = new_label_offset
                    centroids_list[c_id] = X_pca[indices[sub0_mask]].mean(dim=0)
                    centroids_list.append(X_pca[indices[sub1_mask]].mean(dim=0))
                    new_label_offset += 1
                            
        return labels, torch.stack(centroids_list)
    
    def update_clusters_with_mask(self, cluster_labels, new_mask):
        cluster_labels = cluster_labels.to(device)
        new_mask = new_mask.to(device)
        updated_labels = cluster_labels.clone()

        H, W = new_mask.shape
        labels_in_mask = torch.unique(new_mask[new_mask != 0])

        for lbl in labels_in_mask.tolist():
            sel_2d = (new_mask == lbl)
            if not sel_2d.any():
                continue
            new_label_val = int(updated_labels.max().item()) + 1
            updated_labels[sel_2d] = new_label_val

        uniq = torch.unique(updated_labels)
        look = torch.full((int(uniq.max().item()) + 1,), -1, device=device, dtype=torch.long)
        for i, u in enumerate(uniq):
            look[int(u.item())] = i
        remapped = look[updated_labels]
        return remapped, uniq
    
    def process_image_data(self, image_path, latent, text_embeds, steps, image_name, original_mask=None, use_depth=False, use_sam=False):
        data_path = utils.get_route(image_path)
        image = utils.load_image(image_path, (512, 512))
        if steps != 0 and os.path.isfile(data_path):
            data_dict = torch.load(data_path, weights_only=False)
            unet_feature = data_dict['unet_feature'].to(self.device)
            mask = data_dict['mask'].to(self.device)
            depth = data_dict['depth']
            C15 = data_dict.get('C15', latent).to(self.device) 
        else:
            if steps != 0:
                _, latent_list, unet_feature = self.forward_process(
                    latent, text_embeds, steps
                )
                C15 = latent_list[-1]
            else:
                unet_feature = latent
                C15 = latent

            depth = self.get_depth(image_path) if use_depth else None
            mask = self.generate_mask(
                unet_feature,
                image,
                latent,
                depth=depth, 
                image_name=image_name,
                original_mask=original_mask,
                use_sam=use_sam
            )

        result_dict = {
            'image': image_path, 
            'feature': latent,
            'unet_feature': unet_feature,
            'mask': mask,
            'C15': C15,
            'depth': depth
        }

        if steps != 0 and not os.path.isfile(data_path):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            torch.save(result_dict, data_path)

        return result_dict
    
