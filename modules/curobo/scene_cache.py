"""Release obsolete Warp meshes while preserving active collision buffers."""


def mesh_stores(planner):
    """Enumerate the native pool's scene stores without walking solver tensors."""
    pool = planner.pool
    scenes = []
    for context in pool.contexts.values():
        scenes.append(context.planner.scene_collision_checker)
        for name in ['checker', 'closed_checker']:
            checker = getattr(context, name, None)
            if checker is not None:
                scenes.append(checker.scene_model)
    for view in pool.contact_views.values():
        scenes.append(view['checker'].scene_model)
    stores = {}
    for scene in scenes:
        meshes = scene.data.meshes
        if meshes is not None:
            stores[id(meshes)] = meshes
    return list(stores.values())


def prune_inactive_meshes(stores, synchronize):
    """Delete only meshes absent from every active environment in each store.

    Captured graph tensor addresses, enabled obstacles, poses and mesh handles
    remain unchanged. Synchronization must succeed before any handle is freed.
    """
    stores = list({id(store): store for store in stores}.values())
    pending = []
    for store in stores:
        active = {name for env in range(store.num_envs) for name in store.get_names(env)}
        missing = active - store.wp_cache.keys()
        if missing:
            raise RuntimeError('Active collision mesh missing from Warp cache: '+str(missing))
        pending.append((store, set(store.wp_cache)-active))
    removed = sum(len(names) for _, names in pending)
    if removed:
        synchronize()
        for store, names in pending:
            for name in names:
                del store.wp_cache[name]
        synchronize()
    return {'stores': len(stores), 'removed_inactive_meshes': removed,
            'retained_active_meshes': sum(len(store.wp_cache) for store in stores),
            'active_collision_buffers_unchanged': True}
