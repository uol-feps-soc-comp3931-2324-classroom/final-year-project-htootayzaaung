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

function updateVideoFeeds(camera_indices) {
    const mainVideo = document.getElementById("main-video-feed");
    const mainFpsDisplay = document.getElementById("main-fps-display");

    camera_indices.forEach((index, i) => {
        const source = new EventSource(`/video_feed/${index}`);

        source.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === "frame") {
                if (i === 0) {  // Primary feed
                    mainVideo.src = "data:image/jpeg;base64," + data.data;
                } else {  // Secondary feed
                    const secondaryVideo = document.getElementById("secondary-video-feed");
                    if (secondaryVideo) {
                        secondaryVideo.src = "data:image/jpeg;base64," + data.data;
                    }
                }
            } else if (data.type === "fps" && i === 0) {
                mainFpsDisplay.innerText = `FPS: ${data.data}`;
            }
        };

        source.onerror = function() {
            console.error(`Error with event source for camera index ${index}`);
        };
    });
}