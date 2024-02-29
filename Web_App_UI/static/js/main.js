function loadModel() {
    var model_name = document.getElementById('model-select').value;
    $.ajax({
        type: 'POST',
        url: '/load_model',
        data: { model_name: model_name },
        success: function(response) {
            console.log('Model loaded successfully');
        },
        error: function(xhr, status, error) {
            console.error('Error loading model:', error);
        }
    });
}

function updateVideoFeed() {
    var video = document.getElementById('video-feed');
    var fpsDisplay = document.getElementById('fps-display'); // Add an element with this id in your HTML
    var source = new EventSource('/video_feed');

    source.onmessage = function(event) {
        var data = JSON.parse(event.data);
        if (data.type === "frame") {
            video.src = 'data:image/jpeg;base64,' + data.data;
        } else if (data.type === "fps") {
            fpsDisplay.innerText = `FPS: ${data.data}`;
        }
    };
}
