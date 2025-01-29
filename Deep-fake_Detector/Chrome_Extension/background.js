chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'checkDeepfake',
    title: 'Check for Deepfake',
    contexts: ['image'] // Only show the option for images
  });
});

// Listen for the context menu item click event
chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === 'checkDeepfake') {
    const imageUrl = info.srcUrl;

    // Send the image URL to the server for deepfake analysis
    fetch('http://localhost:5001/api/analyze/from_url', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageUrl: imageUrl }) // Send the image URL in JSON body
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      // Log the response for debugging
      console.log('Analysis result:', data);

      // Open the popup
      chrome.windows.create({
        url: chrome.runtime.getURL('popup.html'),
        type: 'popup',
        width: 300,
        height: 200
      }, (window) => {
        // Wait for the popup to load, then send the result
        setTimeout(() => {
          chrome.tabs.sendMessage(window.tabs[0].id, {
            result: data.result,
            error: data.error
          }, (response) => {
            if (chrome.runtime.lastError) {
              console.error('Error sending message to popup:', chrome.runtime.lastError);
            } else {
              console.log('Message sent to popup:', response);
            }
          });
        }, 500); // Adjust the delay if needed
      });
    })
    .catch(error => {
      console.error('Error:', error);
    });
  }
});