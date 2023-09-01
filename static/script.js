
document.getElementById('start_button').addEventListener('click', function() {
    fetch('/start_camera')
        .then(response => response.text())
        .then(data => {
            console.log(data);
            var video = document.getElementById('video_feed');
            video.style.display = 'block';
            video.src = '/video_feed';
        });
});

document.getElementById('stop_button').addEventListener('click', function() {
   fetch('/stop_camera')
   .then(response => response.text())
   .then(data => {
       console.log(data);
       var video = document.getElementById('video_feed');
       video.style.display = 'none';
       video.src = '';
   });
});




