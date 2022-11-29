// Adjust Exposure

float[3] adjust_exposure(float rgbIn[3], float exposureIn)
{
  float rgbOut[3];

  float exposure = pow(2, exposureIn);
  
  rgbOut[0] = rgbIn[0] * exposure;
  rgbOut[1] = rgbIn[1] * exposure;
  rgbOut[2] = rgbIn[2] * exposure;
  
  return rgbOut;
}

void main
(
    input varying float rIn,
    input varying float gIn,
    input varying float bIn,
    input varying float aIn,
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    output varying float aOut,
    input float exposure = 0.0
)
{
    float rgbIn[3] = {rIn, gIn, bIn};

    float rgbOut[3] = adjust_exposure(rgbIn, exposure);
    
    rOut = rgbOut[0];
    gOut = rgbOut[1];
    bOut = rgbOut[2];
    aOut = aIn;
}