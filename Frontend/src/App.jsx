import { useState } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./App.css";
import Applayout from "./Components/Applayout";
import Home from "./Components/Home";
import Login from "./pages/Login";
import REgister from "./pages/REgister";

function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <Applayout />,
      children: [
        {
          path: "/home",
          element: <Home />,
        },
      
      ],
    },

    {
        
          path:"/login",
          element:<Login/>
        
    },

    {
      path:"/register",
      element:<REgister/>
    }
  ]);
  return (
    <>
      <RouterProvider router={router} />
    </>
  );
}

export default App;
