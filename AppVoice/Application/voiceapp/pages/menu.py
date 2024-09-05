import flet as ft


def menu(page):
    btn1 = ft.FilledButton(text='Проверить картавость')
    btn2 = ft.FilledButton(text='Курсы')
    col = ft.Column(controls=[btn1, btn2])
    page.add(ft.Container(
                    content=ft.Column([ft.Row([btn1], alignment=ft.MainAxisAlignment.CENTER,
                                              ),
                                       ft.Row([btn2], alignment=ft.MainAxisAlignment.CENTER,
                                              )], alignment=ft.MainAxisAlignment.CENTER),
                    height=400,
                ),)
