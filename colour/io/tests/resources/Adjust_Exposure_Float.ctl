// Adjust Exposure

void main
(
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    output varying float aOut,
    input varying float rIn,
    input varying float gIn,
    input varying float bIn,
    input varying float aIn = 1.0,
    input float exposure = 0.0
)
{
    rOut = rIn * pow(2, exposure);
    gOut = gIn * pow(2, exposure);
    bOut = bIn * pow(2, exposure);
    aOut = aIn;
}
