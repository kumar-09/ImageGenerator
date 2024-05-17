import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const App = () => {
  const [image, setImage] = useState('');

  const fetchImage = async () => {
    const response = await fetch('http://127.0.0.1:8000/refresh');
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    setImage(imageUrl);
  };

  useEffect(() => {
    fetchImage();
  }, []);

  return (
    <div className="container">
      <h1 className="text-center my-4">AI Generated Image</h1>
      <div className="text-center">
        <img src={image} alt="Generated" className="img-fluid" />
      </div>
      <div className="text-center mt-4">
        <button className="btn btn-primary" onClick={fetchImage}>Refresh</button>
      </div>
    </div>
  );
};

export default App;
