function loadModel() {
    const model_name = document.getElementById('model-select').value;
    $.ajax({
        type: 'POST',
        url: '/load_model',
        data: { model_name: model_name },
        success: function() {
            console.log('Model loaded successfully');
        },
        error: function(xhr, status, error) {
            console.error('Error loading model:', error);
        }
    });
}

function unloadModel() {
    $.ajax({
        type: 'POST',
        url: '/unload_model',
        success: function() {
            console.log('Model unloaded successfully');
        },
        error: function(xhr, status, error) {
            console.error('Error unloading model:', error);
        }
    });
}

function initComprehensiveStats(cameraIndices) {
    const statsBody = document.getElementById("camera-stats-body");

    // Clear existing rows to avoid duplication
    while (statsBody.firstChild) {
        statsBody.removeChild(statsBody.firstChild);
    }

    cameraIndices.forEach((index) => {
        const row = document.createElement("tr");  // Create a new row
        const indexCell = document.createElement("td");
        indexCell.innerText = index;  // Camera index
        row.appendChild(indexCell);  // Add to row

        const fpsCell = document.createElement("td");
        const coverageCell = document.createElement("td");
        const camDimensionCell = document.createElement("td");  // New column for camera dimensions
        const bboxDimensionCell = document.createElement("td");  // New column for bounding box dimensions

        row.appendChild(fpsCell);
        row.appendChild(coverageCell);
        row.appendChild(camDimensionCell);  // Add to row
        row.appendChild(bboxDimensionCell);  // Add to row
        statsBody.appendChild(row);  // Add row to table

        const source = new EventSource(`/video_feed/${index}`);  // EventSource for camera index

        source.onmessage = function(event) {
            const data = JSON.parse(event.data);

            // Handle frame
            if (data.type === "frame") {
                const videoFeed = document.getElementById(`camera-feed-${index}`);  // Correct camera feed
                const img = document.createElement("img");
                img.src = "data:image/jpeg;base64," + data.data;  // Frame image data
                img.alt = `Camera ${index}`;
                img.style.width = "100%";  // Ensure proper display
                img.style.height = "auto";  // Maintain aspect ratio
                videoFeed.innerHTML = "";  // Clear old content
                videoFeed.appendChild(img);
            }

            // Handle FPS updates
            if (data.type === "fps") {
                fpsCell.innerText = `${data.data}`;  // Update FPS
            }

            // Handle object coverage
            if (data.type === "object_coverage") {
                coverageCell.innerText = `${parseFloat(data.data).toFixed(2)}%`;  // Update object coverage
            }

            // Handle camera dimensions
            if (data.type === "camera_dimensions") {
                camDimensionCell.innerText = `${data.data}`;  // Update camera dimensions
            }

            // Handle bounding box dimensions
            if (data.type === "bbox_dimensions") {
                bboxDimensionCell.innerText = `${data.data}`;  // Display all bounding box dimensions
            }
        };

        source.onerror = function() {
            console.error(`Error with EventSource for camera index ${index}`);  // Handle errors
        };
    });
}
