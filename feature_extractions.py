"""
Source: https://github.com/kushalvyas/Bag-of-Visual-Words-Python
This is a combination of helpers.py and bag.py without the modeling.
This py file does the Feature Extraction (SIFT and Bag of Visual Words)

Here is also the code for 3D Color Histogram
Source: https://pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
and sample from TUWEL
"""

import cv2
import numpy as np
import argparse
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt

# Sift + Bovw

class ImageHelpers:
    def __init__(self):
        # cv2.xfeatures2d is older and outdated for SIFT in recent builds
        self.sift_object = cv2.SIFT_create()

    def prepare_image(self, img_data):
            """
            Converts (3, H, W) float tensors (from DataLoader) or (H, W, 3) 
            to (H, W) uint8 grayscale format required by OpenCV SIFT.
            """
            # Case 1: Input is (3, H, W)
            if img_data.ndim == 3 and img_data.shape[0] == 3:
                # Transpose to (Height, Width, Channels) -> [32, 32, 3]
                img = np.transpose(img_data, (1, 2, 0))
                # Rescale [0, 1] to [0, 255] and convert to uint8
                img = (img * 255).astype(np.uint8)
                # Convert RGB to Gray
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
            # Case 2: Input is already (H, W, C) or (H, W), but might be float
            # We ensure it is scaled to 0-255 and uint8
            if img_data.dtype != np.uint8:
                # If values are normalized (0-1), scale them up
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)

            # Case 3: If it is still 3-channel (H, W, 3), convert to Gray
            if img_data.ndim == 3:
                return cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
                    
            return img_data

    def features(self, image):
        # Use the robust prepare_image function
        gray_image = self.prepare_image(image)
        
        # Safety check
        if gray_image is None or gray_image.size == 0:
            return [(), None]
        
        try:
            keypoints, descriptors = self.sift_object.detectAndCompute(gray_image, None)
        except Exception as e:
            print(f"SIFT Error: {e}")
            return [(), None]

        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init=10, random_state=42)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None

    def cluster(self):
        """    
        cluster using KMeans algorithm, 

        """
        print("Clustering features using KMeans...")
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):
        """
        Each cluster denotes a particular visual word 
        Every image can be represeted as a combination of multiple 
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word 

        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images

        """

        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            # Check if descriptor is None before accessing length
            if descriptor_list[i] is None:
                continue
                
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count+j]
                else:
                    idx = kmeans_ret[old_count+j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print("Vocabulary Histogram Generated")

    def formatND(self, l):
        """    
        restructures list into vstack array of shape
        M samples x N features for sklearn

        """        
        # Using np.vstack on the whole list at once is significantly
        # faster than iteratively stacking in a loop.
        
        # Filter out None values or empty arrays to prevent crashes
        cleaned_list = [x for x in l if x is not None and len(x) > 0]
        
        if len(cleaned_list) > 0:
            self.descriptor_vstack = np.vstack(cleaned_list)
        else:
            print("Warning: No descriptors found to stack.")
            self.descriptor_vstack = np.array([])
            
        return self.descriptor_vstack

    def plotHist(self, vocabulary=None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([np.sum(vocabulary[:,h], dtype=np.int32) for h in range(self.n_clusters)])

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.descriptor_list = []

    def compute_train_features(self, train_images): # train_model() in bag.py
        """
        This method contains the entire module 
        required for training the bag of visual words model
        
        1. SIFT Extraction
        2. Clustering (Vocabulary Learning)
        3. Histogram Generation
        """
        self.descriptor_list = []
        
        n_images = len(train_images)
        print(f"Processing {n_images} training images for SIFT extraction...")
        
        # --- PHASE 1: Feature Extraction ---
        for i in range(n_images):
            img = train_images[i]
            
            # ====> SIFT Extraction per image <====
            kp, des = self.im_helper.features(img)
            
            # Convert to float32 for numerical stability and memory efficiency
            if des is not None:
                self.descriptor_list.append(des.astype(np.float32))
            else:
                self.descriptor_list.append(None)
            
            if (i+1) % 1000 == 0:
                print(f"  - Extracted features for {i+1}/{n_images} images")

        # --- PHASE 2: Clustering (Vocabulary Learning) ---
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        
        if len(bov_descriptor_stack) < self.no_clusters:
            print(f"Warning: Not enough descriptors ({len(bov_descriptor_stack)}) for {self.no_clusters} clusters. Reducing clusters.")
            self.bov_helper.kmeans_obj.n_clusters = len(bov_descriptor_stack)
            self.bov_helper.n_clusters = len(bov_descriptor_stack)

        # ====> BoVW Clustering happens here <====
        self.bov_helper.cluster()
        
        # --- PHASE 3: Histogram Generation ---
        # ====> BoVW Histogram creation happens here <====
        self.bov_helper.developVocabulary(n_images=n_images, descriptor_list=self.descriptor_list)

        return self.bov_helper.mega_histogram

    def compute_test_features(self, test_images): # test_model() in bag.py
        """ 
        This is to apply the trained vocabulary.
        
        1. SIFT Extraction
        2. Mapping to existing Clusters (Predict)
        3. Histogram Generation
        """
        if self.bov_helper.kmeans_ret is None:
            raise Exception("Model not trained yet! Run compute_train_features() first.")

        test_descriptor_list = []
        n_images = len(test_images)
        
        print(f"Processing {n_images} test images...")

        # PHASE 1: Feature Extraction 
        for i in range(n_images):
            img = test_images[i]
            # SIFT Extraction per image 
            kp, des = self.im_helper.features(img)
            
            # Convert to float32 for consistency with training
            if des is not None:
                test_descriptor_list.append(des.astype(np.float32))
            else:
                test_descriptor_list.append(None)

        # PHASE 2: Mapping to Vocabulary 
        bov_descriptor_stack = self.bov_helper.formatND(test_descriptor_list)

        # locate nearest clusters for each of the visual word (feature)
        print("Mapping test features to visual vocabulary...")
        # BoVW Prediction (Mapping features to words) 
        test_ret = self.bov_helper.kmeans_obj.predict(bov_descriptor_stack)

        # PHASE 3: Histogram Generation 
        # BoVW Histogram creation 
        self.bov_helper.developVocabulary(n_images=n_images, 
                                          descriptor_list=test_descriptor_list, 
                                          kmeans_ret=test_ret)

        return self.bov_helper.mega_histogram
    

# 3D Color Histogram Feature Extraction

def hist_3d_features(X, bins=8):
    """
    Extract 3D color histogram features (all 3 channels combined).
    Similar to the lecture example using OpenCV.
    
    Args:
        X: numpy array of shape (N, 3, H, W) with values in [0,1]
        bins: number of bins per channel (default: 8, giving 8x8x8=512 features)
    
    Returns:
        numpy array of shape (N, bins^3) with normalized histograms
    """
    feats = []
    for img_chw in X:  # (3,H,W), float in [0,1]
        # Convert to uint8 [0,255]
        img = (img_chw * 255).astype(np.uint8)
        # Convert from (C,H,W) to (H,W,C) for OpenCV
        img_hwc = np.transpose(img, (1, 2, 0))
        # OpenCV expects BGR, but since we compute histogram over all channels
        # the order doesn't matter for the histogram
        # Convert RGB to BGR for consistency with lecture code
        img_bgr = img_hwc[:, :, ::-1].copy()
        # Compute 3D histogram over all three channels simultaneously
        # channels [0,1,2] = B,G,R
        # bins for each channel
        # ranges for each channel [0,256]
        hist_3d = cv2.calcHist([img_bgr], [0, 1, 2], None, 
                               [bins, bins, bins], 
                               [0, 256, 0, 256, 0, 256])
        # Flatten the 3D histogram to 1D feature vector
        f = hist_3d.flatten().astype(np.float32)
        # Normalize (L1 normalization)
        f /= (f.sum() + 1e-8)
        feats.append(f)
    return np.vstack(feats)
