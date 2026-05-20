# remote-sensing
## Project Description

This project focuses on automatic **Land Use and Land Cover (LULC) Classification** for an Egyptian region using USGS Landsat 8 multispectral satellite imagery and machine learning techniques. The main objective is to classify the study area into four major land cover classes: **Water, Vegetation/Agriculture, Urban/Built-up, and Bare Soil/Desert**. The project follows a complete remote sensing processing pipeline starting from satellite image acquisition and preprocessing, passing through feature extraction and supervised classification, and ending with the generation of a full color-coded classification map and area statistics. 

The Landsat 8 image used in this work was downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov?utm_source=chatgpt.com) and cropped to a fixed subset of size 2000 × 2000 pixels covering part of the Nile Delta region in Egypt. The raw Digital Number (DN) values were converted into Top-of-Atmosphere (TOA) reflectance through radiometric calibration using metadata calibration coefficients. 

To improve the separability between land cover classes, several spectral indices were computed and added as additional features. These indices include:

* NDVI for vegetation detection
* NDWI for water body extraction
* NDBI for urban area identification

The final feature stack consisted of seven spectral bands combined with three spectral indices, producing a 10-feature dataset for each pixel. 

Ground truth data was prepared manually using ROI labeling in ENVI software, where labeled samples for the four classes were selected. The labeled dataset was then divided into training and testing sets to train and evaluate multiple machine learning classifiers such as:

* Support Vector Machine (SVM)
* Random Forest (RF)
* K-Nearest Neighbors (KNN)
* Decision Tree
* Neural Network (MLP)

The classifiers were compared based on training and testing accuracy in order to determine the best-performing model. 

Finally, the best classifier was applied to predict the class label for every pixel in the image, generating a complete land cover classification map with different colors representing each class. Area statistics and percentage coverage for each class were also calculated to analyze land distribution within the study area. 

This project demonstrates the importance of remote sensing and machine learning in environmental monitoring, urban planning, agricultural management, and large-scale land cover analysis.
