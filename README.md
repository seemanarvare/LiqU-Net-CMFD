# LiqU-Net-CMFD
LiqU-Net: A Liquid Neural Network-Enhanced U-Net for Efficient Copy-Move Forgery Detection and Localization
This repository contains the implementation of the proposed LiqU-Net, a ResNet50 + Liquid Neural Network (LNN) enhanced U-Net model for copy-move forgery detection and localization.
The model integrates:
-ResNet50 as a backbone for feature extraction,
-LNN (Liquid Neural Network) layers for dynamic and adaptive learning,
-U-Net style decoder for precise pixel-level localization.

## Publicly Available Datasets Used

We have trained and evaluated LiqU-Net on the following publicly available CMFD datasets:

- **CoMoFoD (Copy-Move Forgery Detection Dataset)**  
  👉 [Download CoMoFoD](https://www.vcl.fer.hr/comofod/)

- **IMD (Image Manipulation Dataset)**  
  Provides base images and scripts to generate realistic copy-move forgeries; widely used in benchmarking.  
  👉 [Download IMD](https://www1.cs.fau.de/research/multimedia-security/code/image-manipulation-dataset/)

- **COVERAGE Dataset**  
  Designed to challenge algorithms with visually similar genuine objects and annotated masks.  
  👉 [Download COVERAGE](https://github.com/wenbihan/coverage)

- **GRIP Dataset**  
  *Not publicly available. Can be requested from the original authors*
  Citation: D. Cozzolino, G. Poggi, and L. Verdoliva, “Efficient Dense-Field Copy-Move Forgery Detection,” IEEE Transactions on Information Forensics and Security, vol. 10, no. 11, pp. 2284–2297, Nov. 2015, doi: 10.1109/TIFS.2015.2455334. 

- **ARD Dataset**  
  *Not publicly available. Can be requested from the original authors*
  E. Ardizzone, A. Bruno, and G. Mazzola, “Copy-Move Forgery Detection by Matching Triangles of Keypoints,” IEEE Transactions on Information Forensics and        Security, vol. 10, no. 10, pp. 2084–2094, Oct. 2015, doi: 10.1109/TIFS.2015.2445742. 
