function initComprehensiveStats(cameraIndices) {
    const statsBody = document.getElementById("camera-stats-body");

    // Clear existing rows to avoid duplication or misalignment
    while (statsBody.firstChild) {
        statsBody.removeChild(statsBody.firstChild);
    }

    cameraIndices.forEach((index) => {
        const row = document.createElement("tr");  // Create a new row
        const indexCell = document.createElement("td");
        indexCell.innerText = index;  // Camera index
        row.appendChild(indexCell);  // Add index to the row

        const fpsCell = document.createElement("td");  // FPS cell
        const coverageCell = document.createElement("td");  // Object coverage cell

        row.appendChild(fpsCell);
        row.appendChild(coverageCell);
        statsBody.appendChild(row);  // Add row to the table

        const source = new EventSource(`/video_feed/${index}`);  // EventSource for each camera

        source.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.type === "frame") {
                const videoFeed = document.getElementById(`camera-feed-${index}`);  // Get the correct camera feed
                const img = document.createElement("img");
                img.src = "data:image/jpeg;base64," + data.data;  // Set the video feed
                img.alt = `Camera ${index}`;
                img.style.width = "100%";  // Full width to fit in grid
                img.style.height = "auto";  // Maintain aspect ratio
                videoFeed.innerHTML = "";  // Clear previous content
                videoFeed.appendChild(img);
            }

            if (data.type === "fps") {
                fpsCell.innerText = `${data.data}`;  // Update FPS
            } else if (data.type === "object_coverage") {
                coverageCell.innerText = `${parseFloat(data.data).toFixed(2)}%`;  // Update object coverage
            }
        };

        source.onerror = function() {
            console.error(`Error with EventSource for camera index ${index}`);  // Handle errors
        };
    });
}
