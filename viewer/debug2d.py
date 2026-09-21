"""pygame debug window for the 2D stub (ARCHITECTURE.md 4).

The garden on the left: food-odor haze, shade tree, pond, stink patches, the music, dishes, the player's hand,
and the ducks as circles with a heading line, a retina fan and a mood halo. The night draws in as the
sun goes down. A panel on the right for whichever duck is selected: its drives, its descending
neurons, and a running list of what has just happened to anyone.

Debug only, never a gate check. Click a duck to follow it, click the tree to shake it; the keys
are the player's verbs: P pet, C clap, F feed at the mouse, M music at the mouse, H hat.

With a brain attached (`stub --view --brain`) the panel has something to show; without one the ducks
stand still and only the garden is worth looking at.
"""
import numpy as np
import pygame

from world.fields import DISH_R, DUCK_R, SIZE_M, TREE, daylight

PX = 160  # pixels per metre
PANEL_W = 300
COLORS = [(230, 180, 40), (90, 170, 230), (230, 90, 120), (120, 200, 110), (180, 120, 220)]
DRIVES = ("hunger", "thirst", "fatigue", "sleep_pressure", "boredom")
MOODS = (("fear", (140, 180, 255)), ("anger", (240, 90, 70)), ("joy", (255, 220, 90)),
         ("sorrow", (130, 140, 170)))
TOASTS = 9
RETINA_DEG = 75.0  # half-width of one eye's view, matching body/stub2d/retina.py
EYE_DEG = 55.0


class Viewer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((int(SIZE_M * PX) + PANEL_W, int(SIZE_M * PX)))
        pygame.display.set_caption("micro garden: 2D stub")
        self.font = pygame.font.SysFont("Menlo", 13)
        self.big = pygame.font.SysFont("Menlo", 15, bold=True)
        self.selected = 0
        self.seen = {"eaten": 0, "headbutts": 0, "pets": 0, "sounds": 0}
        self.toasts = []

    def alive(self, on_click=None, on_key=None) -> bool:
        """Handle window events; on_click gets each left click in garden metres, on_key each key's
        name and where the mouse is."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return False
            if e.type == pygame.KEYDOWN and on_key:
                mx, my = pygame.mouse.get_pos()
                on_key(pygame.key.name(e.key), (min(mx / PX, SIZE_M), SIZE_M - my / PX))
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                xy = (e.pos[0] / PX, SIZE_M - e.pos[1] / PX)
                if e.pos[0] < SIZE_M * PX:
                    self.click = xy
                    if on_click:
                        on_click(xy)
        return True

    @staticmethod
    def on_tree(xy) -> bool:
        return np.hypot(xy[0] - TREE[0], xy[1] - TREE[1]) < TREE[2]

    def pick(self, xy, pose) -> None:
        """Select the duck nearest a click, if the click was near one."""
        d = np.linalg.norm(pose[:, :2] - np.asarray(xy, float), axis=1)
        if len(d) and d.min() < 4 * DUCK_R:
            self.selected = int(d.argmin())

    def _px(self, xy):
        return int(xy[0] * PX), int((SIZE_M - xy[1]) * PX)

    def _gather_toasts(self, stub) -> None:
        """One line per new thing that happened, newest last."""
        for key, fmt in (("eaten", "{who} ate"), ("headbutts", "{who} shoved {other}"),
                         ("pets", "{who} was petted"), ("sounds", "{who}: {other}")):
            events = getattr(stub, key, [])
            for e in events[self.seen[key]:]:
                who = stub.names[e[1]].replace("duck-", "")
                other = e[2] if len(e) > 2 else ""
                other = stub.names[other].replace("duck-", "") if isinstance(other, (int, np.integer)) else other
                self.toasts.append((e[0], fmt.format(who=who, other=other)))
            self.seen[key] = len(events)
        self.toasts = self.toasts[-TOASTS:]

    def _garden(self, stub, light) -> None:
        odor = stub.world.odor
        haze = (255 * np.sqrt(odor / max(odor.max(), 1e-9))).astype(np.uint8)
        rgb = np.stack([haze // 3, haze // 2 + 40, haze // 3 + 30], axis=-1)[:, ::-1]  # y up
        rgb = (rgb * (0.25 + 0.75 * light)).astype(np.uint8)  # night draws in
        surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(rgb),
                                            (int(SIZE_M * PX), int(SIZE_M * PX)))
        self.screen.blit(surf, (0, 0))
        pygame.draw.circle(self.screen, (20, 40, 30), self._px(TREE[:2]), int(TREE[2] * PX), 2)
        if stub.world.pond is not None:
            x, y, r = stub.world.pond
            pygame.draw.circle(self.screen, (60, 120, 200), self._px((x, y)), int(r * PX))
        for d in stub.world.danger:
            pygame.draw.circle(self.screen, (120, 80, 40), self._px(d), int(DISH_R * PX))
        for f in stub.world.food:
            pygame.draw.circle(self.screen, (250, 250, 250), self._px(f), int(DISH_R * PX))
        if getattr(stub.world, "music", None) is not None:  # the music, and how far it carries
            at = self._px(stub.world.music)
            for ring, alpha in ((0.25, 140), (0.6, 70), (1.2, 35)):
                halo = pygame.Surface((int(2 * ring * PX) + 4,) * 2, pygame.SRCALPHA)
                pygame.draw.circle(halo, (200, 120, 230, alpha), (int(ring * PX) + 2,) * 2, int(ring * PX), 2)
                self.screen.blit(halo, halo.get_rect(center=at))
            pygame.draw.circle(self.screen, (200, 120, 230), at, int(0.07 * PX))
        if getattr(stub.world, "hand", None) is not None:
            pygame.draw.circle(self.screen, (245, 225, 210), self._px(stub.world.hand), int(0.12 * PX))

    def _duck(self, i, pose, body) -> None:
        x, y, h = pose
        c = COLORS[i % len(COLORS)]
        if body is not None:  # a halo in the colour of whatever it is feeling most
            name, tint = max(MOODS, key=lambda m: float(np.atleast_1d(getattr(body, m[0]))[i]))
            strength = float(np.atleast_1d(getattr(body, name))[i])
            if strength > 0.05:
                halo = pygame.Surface((8 * DUCK_R * PX,) * 2, pygame.SRCALPHA)
                pygame.draw.circle(halo, (*tint, int(90 * strength)), (4 * DUCK_R * PX,) * 2, int(3 * DUCK_R * PX))
                self.screen.blit(halo, halo.get_rect(center=self._px((x, y))))
        for side in (1, -1):  # what each eye can see
            mid = h + side * np.radians(EYE_DEG)
            for edge in (mid - np.radians(RETINA_DEG), mid + np.radians(RETINA_DEG)):
                far = (x + 0.45 * np.cos(edge), y + 0.45 * np.sin(edge))
                pygame.draw.line(self.screen, (*c, 70), self._px((x, y)), self._px(far), 1)
        pygame.draw.circle(self.screen, c, self._px((x, y)), int(DUCK_R * PX))
        if i == self.selected:
            pygame.draw.circle(self.screen, (255, 255, 255), self._px((x, y)), int(DUCK_R * PX) + 3, 2)
        tip = (x + 2 * DUCK_R * np.cos(h), y + 2 * DUCK_R * np.sin(h))
        pygame.draw.line(self.screen, (0, 0, 0), self._px((x, y)), self._px(tip), 2)

    def _bar(self, x, y, w, frac, colour, label) -> None:
        pygame.draw.rect(self.screen, (55, 60, 70), (x, y, w, 11))
        pygame.draw.rect(self.screen, colour, (x, y, int(w * np.clip(frac, 0, 1)), 11))
        self.screen.blit(self.font.render(label, True, (215, 220, 230)), (x + w + 8, y - 2))

    def _panel(self, stub, server, light) -> None:
        left = int(SIZE_M * PX)
        pygame.draw.rect(self.screen, (24, 26, 32), (left, 0, PANEL_W, int(SIZE_M * PX)))
        i = min(self.selected, len(stub.pose) - 1)
        clock = "day" if light > 0.6 else "night" if light < 0.2 else "dusk"
        head = f"{stub.names[i]}   t {stub.t:6.1f} s   {clock}"
        self.screen.blit(self.big.render(head, True, COLORS[i % len(COLORS)]), (left + 12, 10))

        y = 40
        body = getattr(server, "body", None) if server else None
        if body is not None:
            for d in DRIVES:
                self._bar(left + 12, y, 120, float(np.atleast_1d(getattr(body, d))[i]), (200, 150, 70), d)
                y += 17
            y += 6
            if bool(np.atleast_1d(body.asleep)[i]):
                self.screen.blit(self.font.render("asleep", True, (150, 170, 255)), (left + 12, y)); y += 18
            for name, tint in MOODS:
                self._bar(left + 12, y, 120, float(np.atleast_1d(getattr(body, name))[i]), tint, name)
                y += 17
        else:
            self.screen.blit(self.font.render("no brain attached (--brain)", True, (150, 150, 160)), (left + 12, y))
            y += 22

        rates = getattr(getattr(server, "decoder", None), "rates", None)
        if rates is not None:
            y += 8
            self.screen.blit(self.font.render("descending neurons, Hz", True, (215, 220, 230)), (left + 12, y))
            y += 18
            for label, idx in (("forward", 0), ("back", 1), ("steer L", 2), ("steer R", 3),
                               ("giant fiber", 4), ("feed", 5)):
                self._bar(left + 12, y, 120, rates[i][idx] / 5.0, (110, 180, 220), f"{label} {rates[i][idx]:.1f}")
                y += 17

        y += 10
        self.screen.blit(self.font.render("P pet  C clap  F feed  M music  H hat", True, (120, 130, 145)), (left + 12, y))
        y += 22
        self.screen.blit(self.font.render("just happened", True, (215, 220, 230)), (left + 12, y))
        y += 18
        for t, line in self.toasts[::-1]:
            self.screen.blit(self.font.render(f"{t:6.1f}  {line}", True, (170, 180, 195)), (left + 12, y))
            y += 16

    def draw(self, stub, server=None) -> None:
        light = daylight(stub.t)
        if getattr(self, "click", None) is not None:
            self.pick(self.click, stub.pose)
            self.click = None
        self._gather_toasts(stub)
        self._garden(stub, light)
        body = getattr(server, "body", None) if server else None
        for i, pose in enumerate(stub.pose):
            self._duck(i, pose, body)
        self._panel(stub, server, light)
        pygame.display.flip()
