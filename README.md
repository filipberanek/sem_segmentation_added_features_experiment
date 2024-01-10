<h1> Semantic segmentation added features experiment </h1>
<h2>Intro</h2>
As I work for Valeo, we did experiment according: https://woodscape.valeo.com/woodscape/download </br>
I have created multiple networks and tested performance of them. As you could see here: https://github.com/filipberanek/phd_lectures_statistics </br>
In this repository I would like to test, if there will be impact by adding features like:
<ul>
  <li>Canny edges</li>
  <li>Sobel edges</li>
  <li>Sift points</li>
  <li>Harris corner</li>
</ul>
The input into network is then (width, height, n_chanels + extracted features). 
<h2>Structure</h2>
There are two folders:
<ul>
  <li>features_adder</li>
  <li>models</li>
</ul>

<h3>feature_adder</h3>
This folder contains information and displaying features that are extracted as well as python script to extract features. 
<h3>models</h3>
This folder contains part that runs network training and evaluation of results. 


