chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'checkDeepfake',
    title: 'Check for Deepfake',
    contexts: ['image'] // Only show the option for images
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === 'checkDeepfake') {
    const imageUrl = info.srcUrl;

    fetch('http://localhost:5001/api/analyze/from_url', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageUrl: imageUrl })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('Analysis result:', data);

      chrome.windows.create({
        url: chrome.runtime.getURL('popup.html'),
        type: 'popup',
        width: 300,
        height: 200
      }, (window) => {
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
        }, 500);
      });
    })
    .catch(error => {
      console.error('Error:', error);

      chrome.windows.create({
        url: chrome.runtime.getURL('popup.html'),
        type: 'popup',
        width: 300,
        height: 200
      }, (window) => {
        setTimeout(() => {
          chrome.tabs.sendMessage(window.tabs[0].id, {
            error: error.message
          }, (response) => {
            if (chrome.runtime.lastError) {
              console.error('Error sending message to popup:', chrome.runtime.lastError);
            } else {
              console.log('Error message sent to popup:', response);
            }
          });
        }, 500);
      });
    });
  }
});