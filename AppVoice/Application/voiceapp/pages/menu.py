import flet as ft
from .exam import exam

def menu(page):
    page.clean()
    def onclick(e):
        exam(page)


    btn1 = ft.FilledButton(text='Проверить картавость')
    btn2 = ft.FilledButton(text='Курсы', on_click=onclick)
    page.add(ft.Container(
                    content=ft.Column([ft.Row([btn1], alignment=ft.MainAxisAlignment.CENTER,
                                              ),
                                       ft.Row([btn2], alignment=ft.MainAxisAlignment.CENTER,
                                              )], alignment=ft.MainAxisAlignment.CENTER),
                    height=400,
                ),)
