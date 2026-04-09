$ErrorActionPreference = "Stop"

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class Win32 {
  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);
  [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
  public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
  public const uint MOUSEEVENTF_LEFTUP = 0x0004;
  public const uint MOUSEEVENTF_WHEEL = 0x0800;
}

public struct RECT {
  public int Left;
  public int Top;
  public int Right;
  public int Bottom;
}
'@

$artifactDir = Join-Path (Get-Location) "ui_test_artifacts"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
Get-ChildItem $artifactDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$windowTitle = "TransformLab: Animated Linear Transformation Visualizer"

function Get-WindowHandle {
  param([string]$Title)
  $deadline = (Get-Date).AddSeconds(20)
  while ((Get-Date) -lt $deadline) {
    $script:target = [IntPtr]::Zero
    $callback = [Win32+EnumWindowsProc]{
      param($hWnd, $lParam)
      $len = [Win32]::GetWindowTextLength($hWnd)
      if ($len -gt 0 -and [Win32]::IsWindowVisible($hWnd)) {
        $sb = New-Object System.Text.StringBuilder ($len + 1)
        [void][Win32]::GetWindowText($hWnd, $sb, $sb.Capacity)
        if ($sb.ToString() -eq $Title) {
          $script:target = $hWnd
          return $false
        }
      }
      return $true
    }
    [Win32]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null
    if ($script:target -ne [IntPtr]::Zero) {
      return $script:target
    }
    Start-Sleep -Milliseconds 250
  }
  return [IntPtr]::Zero
}

function Click-Point {
  param([int]$X, [int]$Y)
  [void][Win32]::SetCursorPos($X, $Y)
  Start-Sleep -Milliseconds 70
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
  Start-Sleep -Milliseconds 35
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
}

function Capture-Window {
  param([IntPtr]$Handle, [string]$Path)
  $rect = New-Object RECT
  if (-not [Win32]::GetWindowRect($Handle, [ref]$rect)) {
    throw "Could not read window rectangle."
  }
  $width = $rect.Right - $rect.Left
  $height = $rect.Bottom - $rect.Top
  $bmp = New-Object System.Drawing.Bitmap $width, $height
  $graphics = [System.Drawing.Graphics]::FromImage($bmp)
  $graphics.CopyFromScreen(
    (New-Object System.Drawing.Point($rect.Left, $rect.Top)),
    [System.Drawing.Point]::Empty,
    (New-Object System.Drawing.Size($width, $height))
  )
  $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
  $graphics.Dispose()
  $bmp.Dispose()
}

function Find-Control {
  param(
    [System.Windows.Automation.AutomationElement]$Root,
    [string]$Name,
    [System.Windows.Automation.ControlType]$ControlType = $null,
    [int]$Index = 0
  )
  if ($null -eq $ControlType) {
    $condition = New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::NameProperty,
      $Name
    )
  } else {
    $condition = New-Object System.Windows.Automation.AndCondition(
      (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty,
        $Name
      )),
      (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        $ControlType
      ))
    )
  }
  $matches = $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
  if ($matches.Count -le $Index) {
    throw "Missing control '$Name' ($ControlType) index $Index"
  }
  return $matches[$Index]
}

function Find-InvokableControl {
  param(
    [System.Windows.Automation.AutomationElement]$Root,
    [string]$Name
  )

  foreach ($controlType in @(
    [System.Windows.Automation.ControlType]::Button,
    [System.Windows.Automation.ControlType]::CheckBox
  )) {
    try {
      return Find-Control -Root $Root -Name $Name -ControlType $controlType
    } catch {
    }
  }

  throw "Missing invokable control '$Name'."
}

function Invoke-Control {
  param([System.Windows.Automation.AutomationElement]$Element)
  $invokePattern = $null
  if ($Element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$invokePattern)) {
    $invokePattern.Invoke()
    return
  }
  $togglePattern = $null
  if ($Element.TryGetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern, [ref]$togglePattern)) {
    $togglePattern.Toggle()
    return
  }
  $selectionPattern = $null
  if ($Element.TryGetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern, [ref]$selectionPattern)) {
    $selectionPattern.Select()
    return
  }
  throw "Control '$($Element.Current.Name)' is not invokable."
}

function Set-SpinnerValue {
  param([System.Windows.Automation.AutomationElement]$Spinner, [double]$Value)
  $rangePattern = $null
  if ($Spinner.TryGetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern, [ref]$rangePattern)) {
    $rangePattern.SetValue($Value)
    return
  }
  throw "Spinner does not support range values."
}

function Get-Spinners {
  param([System.Windows.Automation.AutomationElement]$Root)
  $condition = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Spinner
  )
  return $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Read-SpinnerValues {
  param([object[]]$Spinners)
  $vals = @()
  foreach ($spinner in $Spinners) {
    $valuePattern = $null
    if (-not $spinner.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$valuePattern)) {
      throw "Could not read spinner value."
    }
    $vals += [double]$valuePattern.Current.Value
  }
  return $vals
}

function Assert-Approx {
  param([double]$Actual, [double]$Expected, [string]$Label, [double]$Tolerance = 0.001)
  if ([Math]::Abs($Actual - $Expected) -gt $Tolerance) {
    throw "Assertion failed for $Label. Expected $Expected but got $Actual"
  }
}

function Drag-And-Zoom {
  param([IntPtr]$Handle)
  $rect = New-Object RECT
  [void][Win32]::GetWindowRect($Handle, [ref]$rect)
  $startX = $rect.Left + 380
  $startY = $rect.Top + 390
  $endX = $startX + 200
  $endY = $startY - 110

  Click-Point -X $startX -Y $startY
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
  Start-Sleep -Milliseconds 60
  [void][Win32]::SetCursorPos($endX, $endY)
  Start-Sleep -Milliseconds 120
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
  Start-Sleep -Milliseconds 100
  [Win32]::mouse_event([Win32]::MOUSEEVENTF_WHEEL, 0, 0, [uint32]240, [UIntPtr]::Zero)
}

function Switch-MatrixSize {
  param(
    [System.Windows.Automation.AutomationElement]$Window,
    [System.Windows.Automation.AutomationElement]$Combo,
    [string]$TargetItemName,
    [string]$ExpectedGroupName
  )
  $groupCondition = New-Object System.Windows.Automation.AndCondition(
    (New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::NameProperty,
      $ExpectedGroupName
    )),
    (New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
      [System.Windows.Automation.ControlType]::Group
    ))
  )
  $itemCondition = New-Object System.Windows.Automation.AndCondition(
    (New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::NameProperty,
      $TargetItemName
    )),
    (New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
      [System.Windows.Automation.ControlType]::ListItem
    ))
  )

  for ($attempt = 0; $attempt -lt 5; $attempt++) {
    $rect = $Combo.Current.BoundingRectangle
    $centerX = [int]($rect.Left + ($rect.Width * 0.50))
    $centerY = [int]($rect.Top + ($rect.Height * 0.52))
    Click-Point -X $centerX -Y $centerY
    Start-Sleep -Milliseconds 180
    [System.Windows.Forms.SendKeys]::SendWait("%{DOWN}")
    Start-Sleep -Milliseconds 180
    if ($TargetItemName -like "3 x 3*") {
      [System.Windows.Forms.SendKeys]::SendWait("{DOWN}{ENTER}")
    } else {
      [System.Windows.Forms.SendKeys]::SendWait("{UP}{ENTER}")
    }

    Start-Sleep -Milliseconds 600
    $group2x2 = $Window.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $groupCondition)
    if ($null -ne $group2x2) {
      return $true
    }
  }

  return $false
}

$results = New-Object System.Collections.Generic.List[string]
$process = Start-Process ".\\.venv\\Scripts\\python.exe" -ArgumentList "main.py" -PassThru

try {
  Start-Sleep -Seconds 8
  $handle = Get-WindowHandle -Title $windowTitle
  if ($handle -eq [IntPtr]::Zero) {
    throw "App window was not found."
  }
  [void][Win32]::SetForegroundWindow($handle)
  $window = [System.Windows.Automation.AutomationElement]::FromHandle($handle)

  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "01_initial.png")
  $results.Add("Launch: PASS")

  $sizeCombo = Find-Control -Root $window -Name "Matrix Size" -ControlType ([System.Windows.Automation.ControlType]::ComboBox)
  $group2x2 = $window.FindFirst(
    [System.Windows.Automation.TreeScope]::Descendants,
    (New-Object System.Windows.Automation.AndCondition(
      (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty,
        "Custom 2 x 2 Matrix"
      )),
      (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Group
      ))
    ))
  )
  if ($null -eq $group2x2) {
    if (-not (Switch-MatrixSize -Window $window -Combo $sizeCombo -TargetItemName "2 x 2 (XY plane)" -ExpectedGroupName "Custom 2 x 2 Matrix")) {
      throw "Could not switch matrix size to 2x2."
    }
  }
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "02_mode_2x2.png")
  $results.Add("Matrix size switch to 2x2: PASS")

  $spinners2 = Get-Spinners -Root $window
  if ($spinners2.Count -ne 4) {
    throw "Expected 4 spinner controls in 2x2 mode, got $($spinners2.Count)"
  }
  foreach ($idx in @(0, 1, 2, 3)) {
    if (-not $spinners2[$idx].Current.IsEnabled) {
      throw "2x2 mode has a disabled visible spinner index $idx."
    }
  }
  $results.Add("2x2 matrix cell enable/disable: PASS")

  $rotationPreset2d = Find-InvokableControl -Root $window -Name "Rotation"
  Invoke-Control -Element $rotationPreset2d
  Start-Sleep -Milliseconds 600
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "03_2x2_rotation_preset.png")
  $results.Add("2x2 rotation preset: PASS")

  $spinnersEdit2d = Get-Spinners -Root $window
  Set-SpinnerValue -Spinner $spinnersEdit2d[0] -Value 1.4
  Set-SpinnerValue -Spinner $spinnersEdit2d[1] -Value 0.3
  Set-SpinnerValue -Spinner $spinnersEdit2d[2] -Value -0.2
  Set-SpinnerValue -Spinner $spinnersEdit2d[3] -Value 0.9
  $applyButton = Find-Control -Root $window -Name "Apply" -ControlType ([System.Windows.Automation.ControlType]::Button)
  Invoke-Control -Element $applyButton
  Start-Sleep -Milliseconds 500
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "04_2x2_custom_apply.png")
  $results.Add("2x2 custom matrix apply: PASS")

  Drag-And-Zoom -Handle $handle
  Start-Sleep -Milliseconds 450
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "05_camera_moved.png")

  $homeButton = Find-Control -Root $window -Name "Home View" -ControlType ([System.Windows.Automation.ControlType]::Button)
  Invoke-Control -Element $homeButton
  Start-Sleep -Milliseconds 500
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "06_home_view.png")
  $results.Add("Home View action: PASS")

  $resetButton = Find-Control -Root $window -Name "Reset All" -ControlType ([System.Windows.Automation.ControlType]::Button)
  Invoke-Control -Element $resetButton
  Start-Sleep -Milliseconds 700
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "07_reset_all.png")

  $group2x2AfterReset = $window.FindFirst(
    [System.Windows.Automation.TreeScope]::Descendants,
    (New-Object System.Windows.Automation.AndCondition(
      (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty,
        "Custom 2 x 2 Matrix"
      )),
      (New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Group
      ))
    ))
  )
  if ($null -eq $group2x2AfterReset) {
    throw "Reset All did not preserve the current 2x2 mode."
  }

  $presetCombo = Find-Control -Root $window -Name "Choose a Transformation" -ControlType ([System.Windows.Automation.ControlType]::ComboBox)
  $presetValuePattern = $null
  [void]$presetCombo.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$presetValuePattern)
  if ($null -eq $presetValuePattern -or $presetValuePattern.Current.Value -ne "Identity") {
    throw "Reset All did not restore Identity preset."
  }

  $spinnersReset = Get-Spinners -Root $window
  $resetVals = Read-SpinnerValues $spinnersReset
  Assert-Approx -Actual $resetVals[0] -Expected 1.0 -Label "reset m11"
  Assert-Approx -Actual $resetVals[3] -Expected 1.0 -Label "reset m22"
  Assert-Approx -Actual $resetVals[1] -Expected 0.0 -Label "reset m12"
  Assert-Approx -Actual $resetVals[2] -Expected 0.0 -Label "reset m21"
  $results.Add("Reset All state restore: PASS")

  if (-not (Switch-MatrixSize -Window $window -Combo $sizeCombo -TargetItemName "3 x 3 (3D extension)" -ExpectedGroupName "Custom 3 x 3 Matrix")) {
    throw "Could not switch matrix size to 3x3."
  }
  Start-Sleep -Milliseconds 700
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "08_mode_3x3.png")
  $results.Add("Matrix size switch to 3x3: PASS")

  $shearPreset3d = Find-InvokableControl -Root $window -Name "Shear"
  Invoke-Control -Element $shearPreset3d
  Start-Sleep -Milliseconds 600
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "09_3x3_shear_preset.png")

  $spinners3d = Get-Spinners -Root $window
  $shearVals = Read-SpinnerValues $spinners3d
  Assert-Approx -Actual $shearVals[1] -Expected 0.8 -Label "3x3 shear m12"
  Assert-Approx -Actual $shearVals[5] -Expected 0.45 -Label "3x3 shear m23"
  Assert-Approx -Actual $shearVals[6] -Expected 0.2 -Label "3x3 shear m31"
  $results.Add("3x3 shear preset: PASS")

  $animateButton = Find-Control -Root $window -Name "Animate" -ControlType ([System.Windows.Automation.ControlType]::Button)
  Invoke-Control -Element $animateButton
  Start-Sleep -Milliseconds 280
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "10_animation_mid.png")
  Start-Sleep -Milliseconds 1050
  Capture-Window -Handle $handle -Path (Join-Path $artifactDir "11_animation_end.png")
  $results.Add("Animation trigger: PASS")

  $results | ForEach-Object { Write-Output $_ }
  Write-Output ("Artifacts: " + $artifactDir)
} finally {
  if ($process -and -not $process.HasExited) {
    Stop-Process -Id $process.Id
  }
}
