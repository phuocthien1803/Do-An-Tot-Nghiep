' Windows TTS helper for TTSEngine (equivalent of macOS "say"/"afplay").
' Usage:
'   cscript //nologo sapi_tts.vbs say <voice_hint> <rate -10..10> <text>
'   cscript //nologo sapi_tts.vbs play <wav_path>
' Voice choice: a token whose name contains <voice_hint>, else the first
' Vietnamese (viVN) voice, else the system default. OneCore voices (e.g.
' "Microsoft An") are searched first because the Vietnamese voice usually
' exists only there. TTSEngine stops playback by terminating this process.
Option Explicit
Dim args, voice, mode
Set args = WScript.Arguments
If args.Count < 2 Then WScript.Quit 2
mode = LCase(args(0))
Set voice = CreateObject("SAPI.SpVoice")

If mode = "play" Then
  Dim stream
  Set stream = CreateObject("SAPI.SpFileStream")
  stream.Open args(1), 0, False
  voice.SpeakStream stream
  stream.Close
  WScript.Quit 0
End If

If mode <> "say" Or args.Count < 4 Then WScript.Quit 2

Dim hint, rate, chosen, vietnamese, regPaths, regPath, category, token
hint = LCase(args(1))
rate = CInt(args(2))
Set chosen = Nothing
Set vietnamese = Nothing
regPaths = Array("HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices", _
                 "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices")
On Error Resume Next
For Each regPath In regPaths
  Set category = CreateObject("SAPI.SpObjectTokenCategory")
  category.SetId regPath, False
  If Err.Number = 0 Then
    For Each token In category.EnumerateTokens
      If chosen Is Nothing And Len(hint) > 0 And InStr(1, LCase(token.GetDescription()), hint, 1) > 0 Then Set chosen = token
      If vietnamese Is Nothing And InStr(1, token.Id, "viVN", 1) > 0 Then Set vietnamese = token
    Next
  End If
  Err.Clear
Next
If chosen Is Nothing Then Set chosen = vietnamese
If Not chosen Is Nothing Then Set voice.Voice = chosen
Err.Clear
On Error GoTo 0

If rate < -10 Then rate = -10
If rate > 10 Then rate = 10
voice.Rate = rate
voice.Speak args(3)
WScript.Quit 0
