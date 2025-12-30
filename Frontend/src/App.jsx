import { useState } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./App.css";
import Applayout from "./Components/Applayout";

import Login from "./pages/Login";
import REgister from "./pages/REgister";

import Icon from "./pages/Icon";
import Home from "./pages/home";
import Search from "./pages/Search";
import Profile from "./pages/Profile";
function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <Applayout />,
      children: [
       
        {
          path: "/icon",
          element: <Icon />,
        },
        {
          path:"/",
          element:<Home/>,
        },
        {
          path:"/search",
          element :<Search/> 
        },
        {
          path:'/profile',
          element : <Profile/>,
        },
      
      ],
    },
    {
      path: "/login",
      element: <Login />,//
    },

    {
      path: "/register",
      element: <REgister />,
    },
  ]);
  return (
    <>
      <RouterProvider router={router} />
    </>
  );
}

export default App;
