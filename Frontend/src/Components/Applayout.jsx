
import React from 'react'
import { Outlet } from 'react-router-dom'


const Applayout = () => {
  return (
 <>
<h1>This is constant</h1>
<Outlet/>
 
 </>
  )
}

export default Applayout