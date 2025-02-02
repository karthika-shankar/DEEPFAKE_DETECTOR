chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Received message in popup:', message);
  const resultElement = document.getElementById('result');
  const confidenceElement = document.getElementById('confidence');

  if (message.error) {
    console.log('Error message:', message.error);
    resultElement.textContent = 'Error';
    confidenceElement.textContent = message.error;
    resultElement.className = '';
  } else if (message.result) {
    if (message.result.error === 'No face detected in the image') {
      console.log('No face detected in the image');
      resultElement.textContent = 'No face detected';
      confidenceElement.textContent = '';
      resultElement.className = '';
    } else {
      console.log('Deepfake analysis result:', message.result);
      resultElement.textContent = message.result.is_deepfake ? 'Fake' : 'Real';
      confidenceElement.textContent = message.result.is_deepfake ? `Confidence: ${message.result.confidence * 100} %` : `Confidence: ${(1 - message.result.confidence) * 100} %`;
      resultElement.className = message.result.is_deepfake ? 'fake' : 'real';
    }
  } else {
    console.log('No result available');
    resultElement.textContent = 'No result available';
    confidenceElement.textContent = '';
    resultElement.className = '';
  }
  sendResponse({status: 'received'});
});