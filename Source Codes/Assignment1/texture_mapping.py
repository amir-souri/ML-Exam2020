import numpy as np
import cv2
import transformations as trans
from record import VideoRecorder, PictureRecorder


def main():
    """Texture mapping application.

    Mouse controls:
        - Use the left mouse button and drag to move mesh vertices.
        - Click an empty area to create a new vertex/point.
        - Right-click a vertex to remove it.

    Keyboard controls:
        - r: Reset the deformation.
        - t: Use the deformed mesh as the new base mesh.
        - o: Toggle drawing the mesh itself.
        - p: Save screenshot in the outputs folder.
        - s: Start/stop recording. Result is saved in the outputs folder.
        - q: Quit the application.
    """
    image = cv2.imread('./inputs/po.jpg')
    gui = MeshGUI(2, 2, image)
    gui.loop()


def dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def triangle_properties(triangle):
    pos = np.min(triangle, axis=0)
    upper_right = np.max(triangle, axis=0)
    size = upper_right - pos
    return np.int32(pos), np.int32(size)


class TriangleMesh:
    """2D triangle mesh.
    """

    def __init__(self, width, height, cols, rows, offset=30):
        """Create a new grid-based triangle mesh.

        :param width: Mesh width.
        :param height: Mesh height.
        :param cols: Number of columns for the mesh grid.
        :param rows: Number of rows for the mesh grid.
        :param offset: Offset mesh from edges.
        """
        self.width = width
        self.height = height
        self.base_grid = self._create_grid(cols, rows, offset)
        self.grid = self.base_grid.copy()

        self.triangles = None
        self.update_indexes = None
        self.update()

        self.observers = []

    def add_observer(self, obj):
        """Observer will be notified on mesh updates.
        :param obj: Observable object (has an _update(indices) method).
        """
        self.observers.append(obj)

    def _dirty(self, indices=None):
        # Notify all observers
        for ob in self.observers:
            ob.update(indices)

    def reset(self):
        """Reset mesh deformation.
        """
        self.grid = self.base_grid.copy()
        self.update()
        self._dirty()

    def switch_base(self):
        """Use deformed mesh as the new base mesh.
        """
        self.base_grid = self.grid.copy()
        self.update()
        self._dirty()

    def triangles_rounded(self):
        return self.triangles.astype(np.int32)

    def add_point(self, x, y):
        """Add the specified point to mesh and _update triangle definitions. Notify observers.
        :param x:
        :param y:
        """
        self.base_grid.append((x, y))
        self.grid.append((x, y))
        self.update()
        self._dirty()

    def delete_point(self, i):
        """Remove point at index i. Notify observers.
        :param i:
        """
        del self.grid[i]
        del self.base_grid[i]
        self.update()
        self._dirty()

    def get_point_idx(self, point):
        """Get index of point in mesh.
        :param point:
        :return: Index.
        """
        return self.grid.index(tuple(point))

    def update(self):
        """Update triangles and point indices.
        """
        self._update_triangles()
        self._update_indices()

    def _update_triangles(self):
        subdiv = cv2.Subdiv2D((0, 0, self.width, self.height))
        subdiv.insert(self.grid)

        tri = subdiv.getTriangleList()
        # 3 points per triangle, 2 coordinates per point
        self.triangles = np.reshape(tri, (tri.shape[0], 3, 2))

    def _update_indices(self):
        indexes = [[self.get_point_idx(p) for p in triangle]
                   for triangle in self.triangles]
        self.indeces = np.array(indexes)

    def get_closest_point(self, x, y):
        """Find closest point. Useful for user interaction.
        :param x:
        :param y:
        :return: Point index, distance.
        """
        best, point = min(enumerate(self.grid),
                          key=lambda v: dist(v[1], (x, y)))
        return best, dist(point, (x, y))

    def move_point(self, i, x, y):
        """Move point at index i to new location and _update relevant triangles. Notify observers of changes to
        specific triangles.
        :param i: Point index.
        :param x: New x-coordinate.
        :param y: New y-coordinate.
        """
        self.grid[i] = (x, y)
        idx = np.argwhere(self.indeces == i)
        self.triangles[idx[:, 0], idx[:, 1]] = [x, y]
        self._dirty(idx[:, 0])

    def _create_grid(self, cols, rows, offset):
        points = []
        for y in range(rows + 1):
            for x in range(cols + 1):
                points.append(
                    (offset // 2 + x * (self.width - offset) // cols,
                     offset // 2 + y * (self.height - offset) // rows)
                )
        return points

    def get_mapping_points(self, i):
        """Return both original and deformed triangle points for a specific index i.
        :param i:
        :return: transformed triangle, original triangle
        """
        idx, triangle = self.indeces[i], self.triangles[i]
        base_triangle = np.array([self.base_grid[i] for i in idx])

        return triangle, base_triangle


class TextureMap:

    def __init__(self, texture, mesh):
        """Initialise texture-map from a texture (image) and a triangle mesh object.
        :param texture:
        :param mesh:
        """
        self.texture = texture
        self.width, self.height = texture.shape[1], texture.shape[0]
        self.mesh = mesh
        self.mesh.add_observer(self)
        self.patches = [None] * len(self.mesh.triangles)
        self.cached = None
        self.update()
        x = len(self.mesh.triangles)

    def update(self, indices=None):
        if indices is None:
            self.cached = [None] * len(self.mesh.triangles)  # nomber of triangles
        else:
            for i in indices:
                self.cached[i] = None

    def _update_patches(self, indices):
        # <Exam 1.3 (3)>
        # for i in indices:
        #     triangle, base_triangle = self.mesh.get_mapping_points(i)
        #     transformed_patch = self._transform_patch(triangle, base_triangle)
        #     mas_patch = self._mask_patch(transformed_patch, triangle)
        #     self.patches[i] = mas_patch

        for i in indices:
            if self.cached[i] is None:
                triangle, base_triangle = self.mesh.get_mapping_points(i)
                transformed_patch = self._transform_patch(triangle, base_triangle)
                mas_patch = self._mask_patch(transformed_patch, triangle)
                self.patches[i] = mas_patch
                self.cached[i] = mas_patch
            else:
                self.patches[i] = self.cached[i]

    def get_transformed(self):
        """Get texture mapped onto the deformed mesh.
        :return: Image of the mapped texture.
        """
        # Create empty list of patches and _update patches for each triangle in the mesh.
        self.patches = [None] * len(self.mesh.triangles)
        self._update_patches(range(len(self.mesh.triangles)))
        # <Exam 1.3 (4)>
        empty = np.zeros(self.texture.shape, dtype=np.uint8)
        for p in self.patches:
            empty = cv2.add(empty, p)
        return empty

    def _transform_patch(self, triangle_transformed, triangle_base):
        # <Exam 1.3 (1)>
        aff = trans.learn_affine(triangle_base, triangle_transformed)
        rows, cols, _ = self.texture.shape
        transformed_image = cv2.warpPerspective(self.texture, aff, (cols, rows))
        return transformed_image

    def _mask_patch(self, patch, triangle):
        # <Exam 1.3 (2)>
        empty = np.zeros(patch.shape[:2], dtype=np.uint8)
        poly_mask = cv2.fillPoly(empty, np.int32([triangle]), 255)
        res = cv2.bitwise_and(patch, patch, mask=poly_mask)
        return res


class MeshGUI:
    """GUI for experimenting with texture mapping."""

    def __init__(self, cols, rows, image, title='MeshGUI', selection_radius=15):
        """Create a new GUI
        :param cols: Columns in initial mesh.
        :param rows: Rows in initial mesh.
        :param image: Image to use as texture for the mapping.
        :param title: Window title.
        :param selection_radius: Mouse selection radius
        """
        self.selection_radius = selection_radius
        self.selected = None
        self.is_dragging = False
        self.draw_structure = True

        # Fix size and crop image to center square
        self.width, self.height = 500, 500
        image = self._center_crop_and_resize(image)
        self.frame = image
        self.display_image = image

        # This will be used for video recording
        self.writer = None

        # Create new mesh and texture-map objects.
        self.mesh = TriangleMesh(self.width, self.height, cols, rows)
        self.texture = TextureMap(self.frame, self.mesh)

        # Initialise window
        self.title = title
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, self._handle_event)

    def _center_crop_and_resize(self, img):
        diff = (img.shape[1] - img.shape[0]) // 2
        img = img[:, diff:img.shape[0] + diff]
        return cv2.resize(img, (self.width, self.height))

    def loop(self):
        while True:
            self._update()

            cv2.imshow(self.title, self.display_image)

            if self.writer is not None:
                self.writer.write(self.display_image)

            k = cv2.waitKey(1)

            if k == ord('q'):
                return
            elif k == ord('r'):
                self.mesh.reset()
            elif k == ord('t'):
                self.mesh.switch_base()
            elif k == ord('o'):
                self.draw_structure = not self.draw_structure
            elif k == ord('p'):
                PictureRecorder('outputs', self.display_image)
            elif k == ord('s'):
                if self.writer is None:
                    self.writer = VideoRecorder('outputs', self.display_image.shape)
                    print('Started recording')
                else:
                    self.writer.stop()
                    self.writer = None
                    print('Finished recording')

    def _update(self):
        """Update texture-map image and draw mesh if active.
        """
        self.display_image = self.texture.get_transformed().copy()

        if self.draw_structure:
            self._update_lines()
            self._update_points()

    def _update_lines(self):
        cv2.polylines(self.display_image,
                      self.mesh.triangles_rounded(), True, (50, 0, 200), 2)

    def _update_points(self):
        for i, p in enumerate(self.mesh.grid):
            if self.selected is not None and i == self.selected:
                cv2.drawMarker(self.display_image, p, (0, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 15, 5)
            else:
                cv2.drawMarker(self.display_image, p, (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 10, 5)

    def _handle_drag_start(self, x, y):
        best, distance = self.mesh.get_closest_point(x, y)
        if distance < self.selection_radius:
            self.selected = best
            self.is_dragging = True
        else:
            self.is_dragging = False
            self.mesh.add_point(x, y)
            self._update()

    def _handle_drag_update(self, x, y):
        if self.is_dragging:
            self.mesh.move_point(self.selected, x, y)
            self._update()
        else:
            best, distance = self.mesh.get_closest_point(x, y)
            if distance < self.selection_radius:
                self.selected = best
            else:
                self.selected = None
            self._update()

    def _handle_drag_end(self):
        self.is_dragging = False

    def _handle_delete_point(self):
        self.mesh.delete_point(self.selected)
        self.selected = None
        self.is_dragging = False

        self._update()

    def _handle_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_drag_start(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._handle_drag_update(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._handle_drag_end()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._handle_delete_point()


if __name__ == '__main__':
    main()