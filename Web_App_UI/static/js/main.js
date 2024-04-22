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

function updateVideoFeeds() {  // Updated function to manage multiple feeds
    const mainVideo = document.getElementById("main-video-feed");
    const secondaryVideo = document.getElementById("secondary-video-feed");
    const mainFpsDisplay = document.getElementById("main-fps-display");

    // Main video feed using camera index 0
    const mainSource = new EventSource("/video_feed/0");

    mainSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
            mainVideo.src = "data:image/jpeg;base64," + data.data;
        } else if (data.type === "fps") {
            mainFpsDisplay.innerText = `FPS: ${data.data}`;
        }
    };

    // Secondary video feed using camera index 4
    const secondarySource = new EventSource("/video_feed/4");

    secondarySource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
            secondaryVideo.src = "data:image/jpeg;base64," + data.data;
        }
    };

    mainSource.onerror = function() {
        console.error("Error with main event source");
    };

    secondarySource.onerror = function() {
        console.error("Error with secondary event source");
    };
}
