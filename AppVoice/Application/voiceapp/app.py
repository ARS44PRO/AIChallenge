import flet as ft
from flet import Page
from .pages.menu import menu

class App:
    def __init__(self, page: Page):
        page.title = 'Картавые'
        page.theme_mode = 'light'
        page.scroll = ft.ScrollMode.AUTO
        page.padding = ft.padding.all(0)
        page.window_width = 400
        page.window_height = 800
        menu(page)
        page.update()
        self.page = page
