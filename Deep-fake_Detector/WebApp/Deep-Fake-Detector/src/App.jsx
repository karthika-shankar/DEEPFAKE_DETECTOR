import Header from "./Header"
import Hero from "./Hero"
import Features from "./Features"
import Upload from "./Upload"
import Footer from "./Footer"
import Body from "./Body"
import { useState } from 'react';



const App = () => {
  const [searchQuery, setSearchQuery] = useState('');

  return(
    <>
    <Body />
    <Header onSearch={setSearchQuery} />
    <Hero/>
    <Features searchQuery={searchQuery} />
    <Footer/>
    </>
  )
}

export default App
