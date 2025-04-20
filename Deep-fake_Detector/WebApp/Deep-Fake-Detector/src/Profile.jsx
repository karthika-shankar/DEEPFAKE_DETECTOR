import React, { useEffect, useState } from 'react';
import { supabase } from './supabaseClient';

const Profile = () => {
  const [user, setUser] = useState(null); // State to store user information
  const [loading, setLoading] = useState(true); // State to handle loading

  useEffect(() => {
    const fetchUser = async () => {
        try {
          // Get the auth token from local storage
          const token = localStorage.getItem('authToken');
          if (!token) {
            throw new Error('No authentication token found.');
          }
  
          // Decode the token to extract the user ID
          const payload = JSON.parse(atob(token.split('.')[1])); // Decode the JWT payload
          const userId = payload.sub; // Extract the user ID (subject)
  
          // Fetch user data from the database
          const { data, error } = await supabase
            .from('user') // Replace 'user' with your table name
            .select('name, email')
            .eq('id', userId)
            .single();;

            if (error) throw error;
            setUser(data); // Set user data
          } catch (error) {
            console.error('Error fetching user data:', error);
          } finally {
            setLoading(false); // Stop loading
          }
        };

    fetchUser();
  }, []);

  if (loading) {
    return <div>Loading...</div>; // Show a loading state
  }

  if (!user) {
    return <div>No user data found.</div>; // Handle case where no user data is found
  }

  return (
    <div className="auth-container">
      <h2>User Profile</h2>
      <p><strong>Name:</strong> {user.name}</p>
      <p><strong>Email:</strong> {user.email}</p>
    </div>
  );
};

export default Profile;