"""Check the installed Warp contract before supplying an equivalent query."""
import re


def function_body(text, signature):
    start = text.index('{', text.index(signature));depth = 1;end = start+1
    while depth:
        depth += (text[end] == '{')-(text[end] == '}');end += 1
    return text[start+1:end-1]


def tokens(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d*)?|[^\s]', text)


def verify_native_contract(header):
    signed = function_body(header, 'mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside')
    unsigned = function_body(header, 'mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist')
    signed = signed.replace('inside = mesh_query_inside_ray_tracing(id, point);', '')
    if tokens(signed) != tokens(unsigned):
        raise ValueError('Signed/unsigned nearest-point traversal differs; no equivalent fast query')
    # Skip the forward declaration and inspect the actual definition.
    signature = 'CUDA_CALLABLE inline float mesh_query_inside_ray_tracing(uint64_t id, const vec3& p)\n{'
    inside = function_body(header, signature)
    for text in ['i < 3', 'mesh_query_ray_closest_sign', 'sign < 0', 'vote >= 2']:
        if text not in inside: raise ValueError('Installed Warp majority semantics changed')
    return {'nearest_point_traversal_identical': True, 'original_axis_order': ['+X', '+Y', '+Z'],
            'original_three_axis_majority': True, 'geometry_and_distance_thresholds_unchanged': True}


def early_vote(votes):
    yes = no = 0
    for used, value in enumerate(votes, 1):
        yes += bool(value);no += not value
        if yes == 2 or no == 2: return yes >= 2, used
    raise ValueError('A complete three-vote decision is required')
