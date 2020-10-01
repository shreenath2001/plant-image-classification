var classify = document.getElementById('classify');
var spinner = document.querySelector('.spinner-border');
var image1 = document.querySelector('#output')
var image2 = document.querySelector('#output-image')
var label = document.querySelector('#label')


var loadFile = function(event) {
    spinner.style.display = "block";
    setTimeout(function(){ spinner.style.display = "none";
    var image = document.getElementById('output');
    image1.style.display = "block";
    image1.style.margin = "auto";
    image.src = URL.createObjectURL(event.target.files[0]);
    classify.style.display = "block"; }, 3000);
};

disp = () => {
    image1.style.display = "none";
    image2.style.display = "none";
    label.style.display = "none";
}