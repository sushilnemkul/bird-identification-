

import Layout from '@/pages/layout'

import React from 'react'
import { Outlet } from 'react-router-dom'


const Applayout = () => {
  return (
 <div>
  <Layout children={<Outlet/>}/>

 </div>
  )
}

export default Applayout