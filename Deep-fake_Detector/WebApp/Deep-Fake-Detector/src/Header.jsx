import React, { useState } from 'react';
import sampleVideo from './assets/stock2.mp4';
import logo from './assets/logo.png';
import Search_icon from './assets/search_icon.png';


const Header = ({ onSearch }) => {
    const [isSearchOpen, setIsSearchOpen] = useState(false);
  
    const toggleSearch = () => {
      setIsSearchOpen(!isSearchOpen);
    };

  
    return (
      <header className="sleek-video-header">
        <video autoPlay loop muted className="background-video">
          <source src={sampleVideo} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="header-content">
            <div className="left-section">
                <div className="logo-container">
                    <img className='logo' src={logo} alt="LOGO" />
                    <a href='https://youtu.be/CgkZ7MvWUAA?si=LmkrEBaOywQxlBSW' className="heading" ><h1>Deepfake Detector</h1></a>
                </div>
            </div>
          <div className="search-container">
            {!isSearchOpen ? (
              <button className="search-icon" onClick={toggleSearch}>
                <img className='Search_icon' src={Search_icon} alt="Search Icon" /> {/* Search Icon */}
              </button>
            ) : (
            <>
              <input
                type="text"
                className="search-input"
                placeholder="Search for features..."
              />
              <button class="clear-btn" onClick={toggleSearch}>X</button>
            </>
            )}
          </div>
        </div>
      </header>
    );
  };
  
  export default Header;