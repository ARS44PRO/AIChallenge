import flet as ft


def exam(page):
    page.clean()
    path = "/Users/nikolay/PycharmProjects/AIChallenge/AppVoice/Application/voiceapp/pages/test-audio-file.wav"

    async def handle_start_recording(e):
        print(f"StartRecording: {path}")
        lab.value = 'Идёт запись'
        page.update()
        await audio_rec.start_recording_async(path)

    async def handle_stop_recording(e):
        output_path = await audio_rec.stop_recording_async()
        print(f"StopRecording: {output_path}")
        lab.value = 'Запись завершена'
        if page.web and output_path is not None:
            await page.launch_url_async(output_path)
    async def handle_state_change(e):
        print(f"State Changed: {e.data}")

    audio_rec = ft.AudioRecorder(
        audio_encoder=ft.AudioEncoder.WAV,
        on_state_changed=handle_state_change,
    )
    page.overlay.append(audio_rec)
    page.update()
    rec = ft.FilledButton("Начать запись", on_click=handle_start_recording)
    lab = ft.Text("")
    stop = ft.ElevatedButton("Остановить запись", on_click=handle_stop_recording)
    check = ft.ElevatedButton("Проверить картавость")
    page.add(ft.Column([ft.Row([rec], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([lab], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([stop], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([check], alignment=ft.MainAxisAlignment.CENTER)],
        alignment=ft.MainAxisAlignment.CENTER, height=400)
        ,
    )