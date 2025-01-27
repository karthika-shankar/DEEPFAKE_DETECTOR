import { useRef } from "react";
import sampleimage from "./assets/image_st2.png";
import samplevideo from "./assets/stock7.mp4";

const Hero = () => {
  const videoRef = useRef(null);

  const handleMouseEnter = () => {
    if (videoRef.current) {
      videoRef.current.play(); // Play the video when hovered
    }
  };

  const handleMouseLeave = () => {
    if (videoRef.current) {
      videoRef.current.pause(); // Pause the video when hover ends
      videoRef.current.currentTime = 0; // Reset to the beginning
    }
  };

  return (
    <section className="hero-section">
      <div className="hero-content">
        <h2>Detect Deepfakes with AI</h2>
        <p>Upload images or videos to analyze for potential manipulation</p>
        <div
          className="media-container"
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          <img src={sampleimage} alt="Hero Image" className="hero-image" />
          <video
            ref={videoRef}
            src={samplevideo}
            className="hero-video"
            muted
            loop
            playsInline
          ></video>
        </div>
      </div>
    </section>
  );
};

export default Hero;
