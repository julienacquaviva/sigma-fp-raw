; Inno Setup script for the Windows installer.
;   iscc /DAppVersion=1.4.0 packaging\windows-installer.iss
; Needs dist\SigmaFpRaw.ofx.bundle (python build.py --targets win64).
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{7B0E5C52-5F5B-4E0C-9C3A-5A1F6D2B8E41}
AppName=Sigma fp RAW (DaVinci Resolve OpenFX plug-in)
AppVersion={#AppVersion}
AppPublisher=Sigma fp RAW
DefaultDirName={commoncf64}\OFX\Plugins\SigmaFpRaw.ofx.bundle
DisableDirPage=yes
DisableProgramGroupPage=yes
DirExistsWarning=no
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=SigmaFpRaw-{#AppVersion}-Windows-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=no
UninstallDisplayName=Sigma fp RAW {#AppVersion} (OpenFX plug-in)

[Messages]
WelcomeLabel2=This installs the Sigma fp RAW plug-in into the OpenFX folder that DaVinci Resolve loads.%n%nClose DaVinci Resolve before you continue.

[InstallDelete]
Type: filesandordirs; Name: "{app}\Contents"

[Files]
Source: "..\dist\SigmaFpRaw.ofx.bundle\Contents\Win64\*"; DestDir: "{app}\Contents\Win64"; Flags: ignoreversion
Source: "..\dist\SigmaFpRaw.ofx.bundle\Contents\Resources\*"; DestDir: "{app}\Contents\Resources"; Flags: ignoreversion
Source: "..\dist\SigmaFpRaw.ofx.bundle\Contents\Info.plist"; DestDir: "{app}\Contents"; Flags: ignoreversion

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
