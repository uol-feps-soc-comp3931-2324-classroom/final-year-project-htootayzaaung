// Function to load the model
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

// Function to update the primary camera
function setPrimaryCamera() {
    const selectedIndex = parseInt(document.getElementById('primary-camera-select').value);
    const cameraIndices = [0, 4]; // Define the list of camera indices
    
    // Swap the indices to set the selected camera as the primary feed
    if (cameraIndices[0] !== selectedIndex) {
        cameraIndices[0] = selectedIndex;
        cameraIndices[1] = cameraIndices.find(i => i !== selectedIndex);
        updateVideoFeeds(cameraIndices); // Refresh the video feeds with the new order
    }
}

// Function to update the video feeds
function updateVideoFeeds() {
    const cameraIndices = [0, 4]; // Default camera indices

    const mainVideo = document.getElementById("main-video-feed");
    const mainFpsDisplay = document.getElementById("main-fps-display");

    const source = new EventSource(`/video_feed/${cameraIndices[0]}`); // Primary feed
    source.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
            mainVideo.src = "data:image/jpeg;base64," + data.data;
        } else if (data.type === "fps") {
            mainFpsDisplay.innerText = `FPS: ${data.data}`;
        }
    };

    if (cameraIndices.length > 1) {
        const secondaryVideo = document.getElementById("secondary-video-feed");
        if (secondaryVideo) {
            const source2 = new EventSource(`/video_feed/${cameraIndices[1]}`); // Secondary feed
            source2.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === "frame") {
                    secondaryVideo.src = "data:image/jpeg;base64," + data.data;
                }
            };
        }
    }
}
