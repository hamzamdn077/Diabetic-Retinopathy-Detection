document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("upload_form").addEventListener("submit", function(event) {
      event.preventDefault(); 
  
      var fileInput = document.getElementById("upload");
      var file = fileInput.files[0];
      var formData = new FormData();
      formData.append("file", file);
  
      fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        document.getElementById("prediction_result").innerText = "Predicted Level of Retinopathy: " + data.prediction;

        // Get the predicted level (assuming it's a string)
        var predictionLevel = data.prediction;
        
        // Define a function to map severity to color
        function getSeverityColor(level) {
          switch (level) {
            case "No_DR":
              return "green";
            case "Mild":
              return "yellow";
            case "Moderate":
              return "orange";
            case "Proliferate_DR":
                return "#FF9900";
            case "Severe":
              return "red";
            default:
              return "black"; // Default color for unexpected values
          }
        }
        
        // Get the color based on prediction level
        var textColor = getSeverityColor(predictionLevel);
        
        // Set the text color of the prediction result
        document.getElementById("prediction_result").style.color = textColor;        var uploaded_image = document.getElementById("uploaded_image");
        uploaded_image.src = URL.createObjectURL(file);
        var processed_image = document.getElementById("processed_image");
        processed_image.src = `data:image/jpeg;base64,${data.processed_image}`; // Decode base64 data
        processed_image.onload = function() {
          URL.revokeObjectURL(processed_image.src);
        };
                document.getElementById("result_container").style.display = "block";
      })
      .catch(error => {
        console.error("Error:", error);
      });
    });
  });
  