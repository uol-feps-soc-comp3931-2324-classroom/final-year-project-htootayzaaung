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

function initVideoFeeds(camera_indices) {
    const mainVideo = document.getElementById("main-video-feed");
    const secondaryVideo = document.getElementById("secondary-video-feed");
    const mainFpsDisplay = document.getElementById("main-fps-display");

    const sources = [];

    // Initialize EventSources for both cameras
    camera_indices.forEach((index, i) => {
        sources[i] = new EventSource(`/video_feed/${index}`);

        sources[i].onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === "frame") {
                if (i === 0) {  // Primary feed
                    mainVideo.src = "data:image/jpeg;base64," + data.data;
                } else {  // Secondary feed
                    secondaryVideo.src = "data:image/jpeg;base64," + data.data;
                }
            } else if (data.type === "fps" && i === 0) {
                mainFpsDisplay.innerText = `FPS: ${data.data}`;
            }
        };

        sources[i].onerror = function() {
            console.error(`Error with event source for camera index ${index}`);
        };
    });

    window.sources = sources; // Keep the sources array accessible
}

function setPrimaryCamera() {
    const primaryIndex = document.getElementById('primary-camera-select').value;

    if (window.sources) {
        // Determine the current primary index and find the new secondary index
        const currentPrimarySource = window.sources[0];
        const newPrimarySource = window.sources.findIndex(source => source.url.includes(`/video_feed/${primaryIndex}`));

        if (newPrimarySource !== 0) { // Swap the sources if different
            // Swap primary and secondary feeds
            const temp = window.sources[0];
            window.sources[0] = window.sources[newPrimarySource];
            window.sources[newPrimarySource] = temp;
        }
    }
}
