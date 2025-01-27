import React, { useState } from 'react';
import logo from './assets/logo.png';
import Search_icon from './assets/search_icon.png';

const Header = ({ onSearch }) => {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    if (typeof onSearch === 'function') {
      onSearch(query);
    }
  };

  const toggleSearch = () => {
    setIsSearchOpen(!isSearchOpen);
    if (!isSearchOpen) {
      setSearchQuery('');
      if (typeof onSearch === 'function') {
        onSearch('');
      }
    }
    onSearch('')
  };

  return (
    <header className="sleek-video-header">
      <div className="header-content">
        <div className="left-section">
          <div className="logo-container">
            <img className="logo" src={logo} alt="LOGO" />
            <h1 className="heading">Deepfake Detector</h1>
          </div>
        </div>
        <div className="search-container">
          {!isSearchOpen ? (
            <button className="search-icon" onClick={toggleSearch}>
              <img className="Search_icon" src={Search_icon} alt="Search Icon" />
            </button>
          ) : (
            <>
              <input
                type="text"
                className="search-input"
                placeholder="Search for features..."
                value={searchQuery}
                onChange={handleSearch}
              />
              <button className="clear-btn" onClick={toggleSearch}>×</button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
