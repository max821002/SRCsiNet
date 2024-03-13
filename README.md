# SRCsiNet
The contents are proposed in the manuscript "Physics-Inspired Deep Learning Anti-Aliasing Framework in Efficient Channel State Feedback".
The arxiv version of the manuscript can be accessed through the link: (TBD)
Authors: Yu-Chien Lin, Yan Xin, Ta-Sung Lee, Charlie (Jianzhong) Zhang, and Zhi Ding.
The project members are from University of California, Davis and Samsung Research America.
All the dataset can be downloaded from the link: https://drive.google.com/drive/folders/1KPIpHV4JNYndbT876caZ6l7XsSzNlO6A?usp=sharing

Dataset naming notions:
SC: subcarrier spacing
BWP: bandwidth part
Predix means the input/output dataset.

Example 1: CSI_SC30KHz_BWP10MHz_8x4UPA
Example 2: TypeIIPrecoder_SC30KHz_BWP10MHz_8x4UPA

Dataset specification:
There is a Base station communicating with 1000 independent UEs.  
Each dataset consists of CSI/precoder of 1000 UEs in 16 TTIs.
The typeII precoder choose 4/16 beams for the case of 32/128 antennas, respectively.
