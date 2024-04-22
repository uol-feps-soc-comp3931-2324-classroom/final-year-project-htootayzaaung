function loadModel() {
    const model_name = document.getElementById('model-select').value;
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
    const video = document.getElementById('video-feed');
    const fpsDisplay = document.getElementById('fps-display');  // Ensure this ID matches with HTML

    const source = new EventSource('/video_feed'); // Start event source to get video feed and FPS

    source.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'frame') { // If data is video frame
            video.src = 'data:image/jpeg;base64,' + data.data; // Update video feed
        } else if (data.type === 'fps') { // If data is FPS
            fpsDisplay.innerText = `FPS: ${data.data}`; // Update FPS display
        }
    };

    source.onerror = function() {
        console.error('Error with event source'); // Handle errors
    };
}
