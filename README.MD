<h1 align="center" style="margin-top: 0px;">Deep Memory: Explicit and Searchable Data Modelling</h1>

<div id="img0" align="center">
    <img src="doc/images/seed_nine.png" height="200" >
    <img src="doc/images/htmp_nine.png" height="200" >
    <img src="doc/images/ambiguous_5.png" height="200" >
    <img src="doc/images/def_not_5.png" height="200" >
</div>
<div id="img1" align="center">
    <img src="doc/images/weight_1.png" height="200" >
    <img src="doc/images/missing_middle_8.png" height="200" >
    <img src="doc/images/3bottom.png" height="200" >
    <img src="doc/images/top3.png" height="200" >
</div>

&emsp;

An investigation into the space complexity of structured image data. We sought to understand explicit representations of images - as decomposed into neighborhoods - and the data science requirements to process, store, and recall such representations. The functional code in this project can capture a representation of an image dataset (I used mnist for testing), and query the stored model for likely representations given an existing predicate.

Multiple libraries and code structures were investigated to prototype methods to store and retreive data in high-dimensional space. Sparse matrix representations and large vectors were essential; the [spotify/annoy](https://github.com/spotify/annoy) project proved to best serve our needs from an implementation perspective. 

Conceptually, this project provides some interesting avenues for further investigation into explicitly-structured data, the capture and storage thereof, and subsequent possible applications. 

