"""pygame debug window for the 2D stub: food-odor heatmap, shade, pond (blue), stink patches (brown),
food (white), ducks with heading lines.

Debug only, never a gate check. Drive bars arrive with physiology in Gate 5.
"""
import numpy as np
import pygame

from world.fields import DISH_R, DUCK_R, SIZE_M, TREE

PX = 160  # pixels per metre
COLORS = [(230, 180, 40), (90, 170, 230), (230, 90, 120), (120, 200, 110), (180, 120, 220)]


class Viewer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((int(SIZE_M * PX),) * 2)
        pygame.display.set_caption("micro garden: 2D stub")

    def alive(self) -> bool:
        return not any(e.type == pygame.QUIT for e in pygame.event.get())

    def _px(self, xy):
        return int(xy[0] * PX), int((SIZE_M - xy[1]) * PX)

    def draw(self, stub) -> None:
        odor = stub.world.odor
        shade = (255 * np.sqrt(odor / max(odor.max(), 1e-9))).astype(np.uint8)
        rgb = np.stack([shade // 3, shade // 2 + 40, shade // 3 + 30], axis=-1)[:, ::-1]  # y up
        surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(rgb), self.screen.get_size())
        self.screen.blit(surf, (0, 0))
        pygame.draw.circle(self.screen, (20, 40, 30), self._px(TREE[:2]), int(TREE[2] * PX), 2)
        if stub.world.pond is not None:
            x, y, r = stub.world.pond
            pygame.draw.circle(self.screen, (60, 120, 200), self._px((x, y)), int(r * PX))
        for d in stub.world.danger:
            pygame.draw.circle(self.screen, (120, 80, 40), self._px(d), int(DISH_R * PX))
        for f in stub.world.food:
            pygame.draw.circle(self.screen, (250, 250, 250), self._px(f), int(DISH_R * PX))
        for i, (x, y, h) in enumerate(stub.pose):
            c = COLORS[i % len(COLORS)]
            pygame.draw.circle(self.screen, c, self._px((x, y)), int(DUCK_R * PX))
            tip = (x + 2 * DUCK_R * np.cos(h), y + 2 * DUCK_R * np.sin(h))
            pygame.draw.line(self.screen, (0, 0, 0), self._px((x, y)), self._px(tip), 2)
        pygame.display.flip()
