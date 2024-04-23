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

function initVideoFeeds(cameraIndices) {
    const mainVideo = document.getElementById("main-video-feed");
    const secondaryVideo = document.getElementById("secondary-video-feed");
    const mainFpsDisplay = document.getElementById("main-fps-display");

    window.sources = [];  // Initialize or clear existing sources
    window.cameraIndices = cameraIndices;  // Store the camera indices globally

    // Initialize EventSources for cameras
    cameraIndices.forEach((index, i) => {
        const source = new EventSource(`/video_feed/${index}`);

        source.onmessage = function(event) {
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

        source.onerror = function() {
            console.error(`Error with event source for camera index ${index}`);
        };

        window.sources.push(source);
    });
}


function setPrimaryCamera() {
    const primaryIndex = parseInt(document.getElementById('primary-camera-select').value);

    // Close current EventSources
    window.sources.forEach(source => source.close());

    // Update sources to new primary and secondary feeds
    const newPrimaryIndex = window.cameraIndices.indexOf(primaryIndex);
    const newCameraIndices = [primaryIndex, ...window.cameraIndices.filter(index => index !== primaryIndex)];

    initVideoFeeds(newCameraIndices);
}

