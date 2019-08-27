# NST
Multi NST

Neural Style Transfer with multiple styles and metric
Based on Image Style Transfer Using Convolutional Neural Networks
Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

NST Presentation.pptx is the presentation file_________


Base algorithm
![content](./pictures/algorithm.png)



Base content image

![content](./results/content.png)

Base style image

![style](./results/style.png)


These are the examples of different content matrix and style distance metric(Style loss)

![gram](./results/gram_matrix.png)

![channel](./results/channel-wise_mean_matrix.png)

![js](./results/js_divergence.png)

![minskowki](./results/minskowki_distance.png)



With multiple styles

![4style](./results/4styles.png)

![4style result](./results/4styles_result.png)


With pearson positive and negative distance

![style](./results/noise_style.png)

![style](./results/noise_positive.png)

![style](./results/noise_negative.png)

You can see the image is tring to avoid noise pattern

![style](./results/rectangle_style.png)

![style](./results/rectangle_negative.png)


