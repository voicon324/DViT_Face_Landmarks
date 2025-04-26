import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.heatmap_utils import generate_ground_truth_heatmaps # Assuming utils package is accessible

# Class FaceLandmark300WDataset remains the same as in the notebook
# Make sure imports are correct (e.g., generate_ground_truth_heatmaps)
class FaceLandmark300WDataset(Dataset):
    def __init__(self, root_dir, file_list_path, img_size=256, feature_map_size=(32, 32), sigma=1.5, transform=None, num_landmarks=68):
        """
        Args:
            root_dir (string): Path to the root directory containing images and .pts files.
            file_list_path (string): Path to the text file listing the basenames.
            img_size (int): Image size after resizing.
            feature_map_size (tuple): Heatmap size (H, W).
            sigma (float): Standard deviation for Gaussian heatmap.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_landmarks (int): Number of landmarks (fixed to 68 for 300W).
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.feature_map_size = feature_map_size
        self.sigma = sigma
        self.transform = transform
        self.num_landmarks = num_landmarks # Store this

        self.image_files = []
        self.landmark_files = []
        self.basenames = []

        print(f"Loading 300W data from: {root_dir}")
        print(f"Using file list: {file_list_path}")

        try:
            with open(file_list_path, 'r') as f:
                file_basenames = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File list not found at {file_list_path}")
            raise

        for basename in file_basenames:
            img_path_png = os.path.join(root_dir, basename + '.png')
            img_path_jpg = os.path.join(root_dir, basename + '.jpg')
            pts_path = os.path.join(root_dir, basename + '.pts')

            img_path = None
            if os.path.exists(img_path_png):
                img_path = img_path_png
            elif os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            else:
                 # print(f"Warning: Image file not found for basename {basename} (checked .png and .jpg). Skipping.")
                 continue # Skip if image doesn't exist

            if os.path.exists(pts_path):
                # Check if pts file is parseable before adding
                if self._check_pts_parseable(pts_path):
                    self.image_files.append(img_path)
                    self.landmark_files.append(pts_path)
                    self.basenames.append(basename)
                else:
                    print(f"Warning: Skipping {basename} due to parsing issues in {pts_path}.")
            else:
                # print(f"Warning: Landmark file not found: {pts_path}. Skipping corresponding image {img_path}.")
                pass # Skip if pts doesn't exist

        print(f"Found {len(self.image_files)} valid samples.")
        if len(self.image_files) == 0:
             raise ValueError("No valid image/landmark pairs found. Check root_dir and file_list_path.")


    def __len__(self):
        return len(self.image_files)

    def _check_pts_parseable(self, file_path):
        """Checks if a .pts file seems parseable without fully parsing."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < 4: return False # Basic check for header + {} + points
                if 'version:' not in lines[0] or 'n_points:' not in lines[1] or lines[2].strip() != '{' or lines[-1].strip() != '}':
                    # print(f"Warning: Malformed header/footer in {file_path}") # Optional warning
                    return False # Header/footer format check
                # Check number of points matches header (optional but good)
                try:
                    num_pts_header = int(lines[1].split(':')[1].strip())
                    actual_num_pts = len(lines) - 4 # version, n_points, {, }
                    if num_pts_header != actual_num_pts:
                        # print(f"Warning: Point count mismatch in {file_path}. Header: {num_pts_header}, Actual: {actual_num_pts}") # Optional
                        return False
                    # Ensure expected number matches (if fixed)
                    if num_pts_header != self.num_landmarks:
                         # print(f"Warning: Expected {self.num_landmarks} landmarks in {file_path}, header says {num_pts_header}.") # Optional
                         return False
                except (IndexError, ValueError):
                    # print(f"Warning: Could not parse n_points in {file_path}") # Optional
                    return False
            return True
        except Exception:
            # print(f"Warning: Error reading {file_path}") # Optional
            return False

    def _parse_pts_file(self, file_path):
        """Đọc tọa độ từ file .pts"""
        landmarks = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Bỏ qua header (ví dụ: "version: 1", "n_points: 68", "{")
            point_lines = lines[lines.index('{\n')+1 : -1] # Lấy các dòng giữa { và }
            if len(point_lines) != self.num_landmarks:
                 print(f"Warning: Expected {self.num_landmarks} landmarks in {file_path}, but found {len(point_lines)}. Check file format.")
                 # Có thể return None hoặc xử lý lỗi khác
                 return None # Trả về None nếu số landmark không đúng
    
            for line in point_lines:
                coords = line.strip().split()
                if len(coords) == 2:
                    landmarks.append([float(coords[0]), float(coords[1])]) # x, y
                else:
                     print(f"Warning: Invalid coordinate format in {file_path}: '{line.strip()}'. Skipping this landmark.")
                     # Có thể cần xử lý lỗi chặt chẽ hơn
                     landmarks.append([0.0, 0.0]) # Tạm thời gán giá trị mặc định
    
        return np.array(landmarks, dtype=np.float32)


    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pts_path = self.landmark_files[idx]
        basename = self.basenames[idx]

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"cv2.imread failed for {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_h, original_w = image.shape[:2]

            landmarks = self._parse_pts_file(pts_path)
            if landmarks is None:
                 raise ValueError(f"Failed to parse landmarks from {pts_path}")
            # Ensure correct number of landmarks read, handle potential None from _parse_pts_file
            if landmarks.shape != (self.num_landmarks, 2):
                raise ValueError(f"Incorrect landmark shape {landmarks.shape} parsed from {pts_path}, expected ({self.num_landmarks}, 2)")


            # --- Preprocessing and Augmentation ---
            scale_x = self.img_size / original_w
            scale_y = self.img_size / original_h
            # Use INTER_LINEAR for resizing images usually
            image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # Scale landmarks to the resized image dimensions
            landmarks_img_scale_gt = landmarks * np.array([scale_x, scale_y], dtype=np.float32)

            image_final = image_resized
            landmarks_final = landmarks_img_scale_gt.copy() # Start with scaled landmarks

            if self.transform:
                keypoints_list = landmarks_final.tolist()
                try:
                    # Check if keypoints list is empty before passing
                    if not keypoints_list:
                         raise ValueError("Landmark list is empty before augmentation.")

                    augmented = self.transform(image=image_final, keypoints=keypoints_list)
                    image_final = augmented['image'] # Should be tensor if ToTensorV2 is last
                    augmented_keypoints = augmented['keypoints']

                    # Crucial check: Albumentations might remove keypoints if they go out of bounds
                    # We need to handle this to maintain the fixed number of landmarks.
                    # Option 1: Revert to unaugmented if keypoints are lost (simpler)
                    # Option 2: Pad lost keypoints (more complex, might introduce noise)
                    if len(augmented_keypoints) != self.num_landmarks:
                        # print(f"Warning: {self.num_landmarks - len(augmented_keypoints)} landmarks lost during augmentation for {basename}. Reverting landmarks.")
                        # Revert landmarks, but keep augmented image (common practice)
                        landmarks_final = landmarks_img_scale_gt.copy()
                        # Ensure image_final is still a tensor if transform normally does that
                        if not isinstance(image_final, torch.Tensor):
                             # Re-apply minimal transform if needed (e.g., Normalize + ToTensorV2)
                             # This assumes your transform usually includes these.
                             basic_transform = A.Compose([
                                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ToTensorV2()])
                             image_final = basic_transform(image=augmented['image'])['image'] # Use the geometrically augmented image

                    else:
                         landmarks_final = np.array(augmented_keypoints, dtype=np.float32)


                except Exception as e:
                    print(f"Error during augmentation for {basename}: {e}. Using unaugmented data.")
                    # Fallback to unaugmented, ensuring tensor format
                    if not isinstance(image_final, torch.Tensor):
                        basic_transform = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
                        image_final = basic_transform(image=image_resized)['image']
                    landmarks_final = landmarks_img_scale_gt.copy()

            # Ensure image is a tensor if not already handled by transform
            if not isinstance(image_final, torch.Tensor):
                 basic_transform = A.Compose([
                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ToTensorV2()])
                 image_tensor = basic_transform(image=image_final)['image']
            else:
                 image_tensor = image_final

            # --- Prepare outputs ---
            # 1. Landmarks scaled to feature map size for coordinate loss
            scale_feat_x = self.feature_map_size[1] / self.img_size
            scale_feat_y = self.feature_map_size[0] / self.img_size
            landmarks_for_coord_loss = torch.from_numpy(landmarks_final * np.array([scale_feat_x, scale_feat_y], dtype=np.float32))

            # 2. Ground truth heatmaps generated from feature map scaled landmarks
            # Ensure landmark coordinates are within heatmap bounds before generating
            landmarks_clipped = landmarks_for_coord_loss.clone()
            landmarks_clipped[:, 0].clamp_(0, self.feature_map_size[1] - 1)
            landmarks_clipped[:, 1].clamp_(0, self.feature_map_size[0] - 1)
            gt_heatmaps = generate_ground_truth_heatmaps(landmarks_clipped, self.feature_map_size, self.sigma)

            # 3. Ground truth landmarks at image scale (before augmentation) for evaluation
            landmarks_gt_img_scale_tensor = torch.from_numpy(landmarks_img_scale_gt)

            # Final check on tensor shapes before returning
            if image_tensor.shape != (3, self.img_size, self.img_size) or \
               landmarks_for_coord_loss.shape != (self.num_landmarks, 2) or \
               gt_heatmaps.shape != (self.num_landmarks, self.feature_map_size[0], self.feature_map_size[1]) or \
               landmarks_gt_img_scale_tensor.shape != (self.num_landmarks, 2):
                raise ValueError(f"Shape mismatch for {basename}. Img: {image_tensor.shape}, CoordLmk: {landmarks_for_coord_loss.shape}, Heatmap: {gt_heatmaps.shape}, EvalLmk: {landmarks_gt_img_scale_tensor.shape}")

            return image_tensor, landmarks_for_coord_loss, gt_heatmaps, landmarks_gt_img_scale_tensor, basename

        except Exception as e:
            print(f"CRITICAL ERROR processing index {idx} ({basename}): {e}. Returning dummy data.")
            # Return dummy data of correct shapes and types to avoid crashing the DataLoader
            dummy_image = torch.randn(3, self.img_size, self.img_size, dtype=torch.float32)
            dummy_coords = torch.zeros(self.num_landmarks, 2, dtype=torch.float32)
            dummy_heatmaps = torch.zeros(self.num_landmarks, self.feature_map_size[0], self.feature_map_size[1], dtype=torch.float32)
            dummy_gt_landmarks = torch.zeros(self.num_landmarks, 2, dtype=torch.float32)
            return dummy_image, dummy_coords, dummy_heatmaps, dummy_gt_landmarks, f"error_{basename}"