"use client"

import { useState } from "react"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"

export default function EnhancementOptions() {
  const [options, setOptions] = useState({
    hdrEnhancement: true,
    perspectiveCorrection: true,
    virtualStaging: false,
    skyReplacement: false,
    lawnEnhancement: false,
    objectRemoval: false,
    upscaling: true,
    enhancementLevel: 75,
    skyOption: "blue",
    stagingStyle: "modern",
  })

  const handleSwitchChange = (key: string) => {
    setOptions((prev) => ({
      ...prev,
      [key]: !prev[key as keyof typeof prev],
    }))
  }

  const handleSliderChange = (value: number[]) => {
    setOptions((prev) => ({
      ...prev,
      enhancementLevel: value[0],
    }))
  }

  const handleRadioChange = (key: string, value: string) => {
    setOptions((prev) => ({
      ...prev,
      [key]: value,
    }))
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="hdr-enhancement">HDR Enhancement</Label>
            <p className="text-xs text-gray-500">Fix poor lighting conditions</p>
          </div>
          <Switch
            id="hdr-enhancement"
            checked={options.hdrEnhancement}
            onCheckedChange={() => handleSwitchChange("hdrEnhancement")}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="perspective-correction">Perspective Correction</Label>
            <p className="text-xs text-gray-500">Fix distorted room angles</p>
          </div>
          <Switch
            id="perspective-correction"
            checked={options.perspectiveCorrection}
            onCheckedChange={() => handleSwitchChange("perspectiveCorrection")}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="virtual-staging">Virtual Staging</Label>
            <p className="text-xs text-gray-500">Add furniture to empty rooms</p>
          </div>
          <Switch
            id="virtual-staging"
            checked={options.virtualStaging}
            onCheckedChange={() => handleSwitchChange("virtualStaging")}
          />
        </div>

        {options.virtualStaging && (
          <div className="ml-6 mt-2 border-l-2 pl-4 border-gray-200">
            <Label className="text-sm mb-2 block">Staging Style</Label>
            <RadioGroup
              value={options.stagingStyle}
              onValueChange={(value) => handleRadioChange("stagingStyle", value)}
              className="flex flex-col space-y-1"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="modern" id="modern" />
                <Label htmlFor="modern" className="text-sm">
                  Modern
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="traditional" id="traditional" />
                <Label htmlFor="traditional" className="text-sm">
                  Traditional
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="minimalist" id="minimalist" />
                <Label htmlFor="minimalist" className="text-sm">
                  Minimalist
                </Label>
              </div>
            </RadioGroup>
          </div>
        )}

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="sky-replacement">Sky Replacement</Label>
            <p className="text-xs text-gray-500">Replace dull skies in exterior shots</p>
          </div>
          <Switch
            id="sky-replacement"
            checked={options.skyReplacement}
            onCheckedChange={() => handleSwitchChange("skyReplacement")}
          />
        </div>

        {options.skyReplacement && (
          <div className="ml-6 mt-2 border-l-2 pl-4 border-gray-200">
            <Label className="text-sm mb-2 block">Sky Option</Label>
            <RadioGroup
              value={options.skyOption}
              onValueChange={(value) => handleRadioChange("skyOption", value)}
              className="flex flex-col space-y-1"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="blue" id="blue" />
                <Label htmlFor="blue" className="text-sm">
                  Clear Blue
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="sunset" id="sunset" />
                <Label htmlFor="sunset" className="text-sm">
                  Sunset
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="cloudy" id="cloudy" />
                <Label htmlFor="cloudy" className="text-sm">
                  Partly Cloudy
                </Label>
              </div>
            </RadioGroup>
          </div>
        )}

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="lawn-enhancement">Lawn Enhancement</Label>
            <p className="text-xs text-gray-500">Make lawns greener and healthier</p>
          </div>
          <Switch
            id="lawn-enhancement"
            checked={options.lawnEnhancement}
            onCheckedChange={() => handleSwitchChange("lawnEnhancement")}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="object-removal">Object Removal</Label>
            <p className="text-xs text-gray-500">Remove unwanted objects</p>
          </div>
          <Switch
            id="object-removal"
            checked={options.objectRemoval}
            onCheckedChange={() => handleSwitchChange("objectRemoval")}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="upscaling">Resolution Upscaling</Label>
            <p className="text-xs text-gray-500">Improve resolution of low-quality images</p>
          </div>
          <Switch id="upscaling" checked={options.upscaling} onCheckedChange={() => handleSwitchChange("upscaling")} />
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between">
          <Label htmlFor="enhancement-level">Enhancement Level</Label>
          <span className="text-sm">{options.enhancementLevel}%</span>
        </div>
        <Slider
          id="enhancement-level"
          min={0}
          max={100}
          step={5}
          value={[options.enhancementLevel]}
          onValueChange={handleSliderChange}
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>Subtle</span>
          <span>Balanced</span>
          <span>Dramatic</span>
        </div>
      </div>
    </div>
  )
}
