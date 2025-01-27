import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Logo_r from './assets/Logo_r.png'
import Logo_f from './assets/Logo_f.png'
import Logo from './assets/logo.png'

const favicon = document.querySelector("link[rel='icon']");

const FeatureCard = ({ icon, title, description, onSelect }) => (
  <div className="feature-card" onClick={onSelect}>
    <div className="icon">{icon}</div>
    <h3>{title}</h3>
    <p>{description}</p>
  </div>
);

const ImageIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <polyline points="21 15 16 10 5 21" />
  </svg>
);

const VideoIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="23 7 16 12 23 17 23 7" />
    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
  </svg>
);

const Features = ({ searchQuery }) => {
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const features = [
    {
      id: 'image',
      icon: <ImageIcon />,
      title: 'Image Analysis',
      description: 'Advanced AI detection for manipulated images',
      acceptedTypes: 'image/*',
    },
    {
      id: 'video',
      icon: <VideoIcon />,
      title: 'Video Detection',
      description: 'Frame-by-frame analysis of video content',
      acceptedTypes: 'video/*',
    },
  ];

  const handleDrop = async (acceptedFiles) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const file = acceptedFiles[0];
      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/analyze/image', {
        method: 'POST',
        body: formData
      });
      
  
      // Check for empty response
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
  
      // Try parsing the response
      const data = JSON.parse(text);
      
      if (!response.ok) {
        throw new Error(data.error || 'Server error occurred');
      }
      
      setAnalysisResult(data.result);
      
    } catch (error) {
    console.error('Upload failed:', error);
    setError(error.message);
  } finally {
    setIsLoading(false);
  }
};
  
  
  
const UploadModal = ({ onClose, acceptedTypes }) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    accept: selectedFeature.id === 'image' 
      ? {
          'image/jpeg': ['.jpg', '.jpeg'],
          'image/png': ['.png']
        }
      : {
          'video/mp4': ['.mp4'],
          'video/x-msvideo': ['.avi']
        }
  });

  return (
    <div className="upload-modal">
      <div className="modal-content">
        <h3>Upload For {selectedFeature.title}</h3>
        <div
          {...getRootProps({
            className: `dropzone ${isDragActive ? 'drag-active' : ''}`,
          })}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the files here...</p>
          ) : (
            <p>Drag & drop files here, or click to select files</p>
          )}
        </div>
        
        {isLoading && <p className="loading">Analyzing...</p>}
        {error && <p className="error">{error}</p>}
        
        {imagePreview && (
          <div className="preview">
            <img src={imagePreview} alt="Preview" />
          </div>
        )}
        
        {analysisResult && (
          <div className="result">
            <h4>Analysis Result:</h4>
            {analysisResult.is_deepfake ? <p>The Image Is Fake</p> : <p>The Image Is Real</p>}
            {analysisResult.is_deepfake ? document.title="Fake" : document.title="Real"}
            {analysisResult.is_deepfake ? document.title.li : document.title="Real"}
            {favicon.href = analysisResult.is_deepfake ? Logo_f : Logo_r }
            <p>Confidence: {analysisResult.is_deepfake ? 
                            (analysisResult.confidence * 100).toFixed(2) : 
                            (100 - (analysisResult.confidence * 100)).toFixed(2)}%
            </p>
          </div>
        )}
        
        <button 
          className='Drop-close' 
          onClick={() => {
            onClose();
            setImagePreview(null);
            URL.revokeObjectURL(imagePreview); // Clean up
            favicon.href = Logo;
            document.title="Deep Fake Detector"
          }}
        >
          Close
        </button>
      </div>
    </div>
  );
};


  const filteredFeatures = features.filter((feature) =>
    feature.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <section className="features-section">
      <div className="features-grid">
        {filteredFeatures.map((feature) => (
          <FeatureCard
            key={feature.id}
            {...feature}
            onSelect={() => setSelectedFeature(feature)}
          />
        ))}
      </div>

      {selectedFeature && (
        <UploadModal
          onClose={() => {
            setSelectedFeature(null);
            setAnalysisResult(null);
            setError(null);
          }}
          acceptedTypes={selectedFeature.acceptedTypes}
        />
      )}
    </section>
  );
};

export default Features;