@echo off
REM Script to generate SUMO network from node and edge files

echo Generating SUMO network...

netconvert --node-files=downtown.nod.xml --edge-files=downtown.edg.xml --output-file=downtown.net.xml --tls.guess=true --tls.default-type=actuated --junctions.corner-detail=5 --crossings.guess=true --walkingareas=true

if %ERRORLEVEL% EQU 0 (
    echo Network generated successfully: downtown.net.xml
) else (
    echo Error generating network!
)

pause
