import * as React from "react"

import { Card, CardContent } from "@/components/ui/card"
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel"

import Autoplay from "embla-carousel-autoplay"

export function CarouselSize() {
  const plugin = React.useRef(
    Autoplay({ delay: 3000, stopOnInteraction: false })
  )

  const images = [
    "/public/bird1.jpg",
    "/images/bird2.jpg",
    "/images/bird3.jpg",
    "/images/bird4.jpg",
  ]

  return ( 
    <Carousel
      plugins={[plugin.current]}
      opts={{ align: "start", loop: true }}
      className="w-[1265px] h-[500px] pl-3.5 pr-3.5 rounded-lg overflow-hidden bg-gray-200"
    >
      <CarouselContent>
        {images.map((img, index) => (
          <CarouselItem key={index} className="w-full h-full">
            <Card className="w-full h-[500px]">
              <CardContent className="w-full h-full p-0">
                <img
                  src={img}
                  alt=""
                  className="w-full h-full object-cover"
                />
              </CardContent>
            </Card>
          </CarouselItem>
        ))}
      </CarouselContent>

      <CarouselPrevious />
      <CarouselNext />
    </Carousel>
  )
}
