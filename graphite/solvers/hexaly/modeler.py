import ctypes as _c
from .optimizer import _hx_define_signal_handler
from .optimizer import _encode_string
from .optimizer import _encode_strings_no_null
from .optimizer import _read_string
from .optimizer import HxError
from .optimizer import HxErrorCode
from .optimizer import HxExpression
from .optimizer import _is_string
import numbers
import sys
from .optimizer import HexalyOptimizer
from .optimizer import _encode_string_no_null
from .optimizer import _decode_subset_bytes
from .optimizer import _hx_interrupt
from .optimizer import _hx_set_pending_error
from .optimizer import _lhx
from .optimizer import _hx_check_errors

########


__all__ = [
    "HexalyModeler", "HxmModule", "HxmValue", "HxmFunction", "HxmMap", "HxmReferenceScope"
]


########


class _hxmdata_val(_c.Union):
    _fields_ = [('ref', _c.c_void_p),
                ('intValue', _c.c_longlong),
                ('dblValue', _c.c_double)]


class _hxmdata(_c.Structure):
    _anonymous_ = ('val',)
    _fields_ = [('val', _hxmdata_val),
                ('type', _c.c_size_t)]

class _hxm_ref_guard(object):
    __slots__ = "_ref",

    def __init__(self, ref):
        self._ref = ref

    def __enter__(self):
        _hxm_inc_ref(self._ref)

    def __exit__(self, _1, _2, _3):
        _hxm_dec_ref(self._ref)

_hxm_create_modeler = _lhx.hxm_create_modeler
_hxm_create_modeler.argtypes = []
_hxm_create_modeler.restype = _c.c_void_p
_hxm_create_modeler.errcheck = _hx_check_errors

_hxm_delete_modeler = _lhx.hxm_delete_modeler
_hxm_delete_modeler.argtypes = [_c.c_void_p]
_hxm_delete_modeler.restype = None
_hxm_delete_modeler.errcheck = _hx_check_errors

_hxm_stream_writer_type = _c.CFUNCTYPE(None, _c.c_void_p, _c.c_char_p, _c.c_int, _c.c_void_p)
_hxm_stream_flusher_type = _c.CFUNCTYPE(None, _c.c_void_p, _c.c_void_p)
_hxm_set_std_stream = _lhx.hxm_set_std_stream
_hxm_set_std_stream.argtypes = [_c.c_void_p, _c.c_int, _hxm_stream_writer_type, _hxm_stream_flusher_type, _c.c_void_p]
_hxm_set_std_stream.restype = None
_hxm_set_std_stream.errcheck = _hx_check_errors


_hxm_create_module = _lhx.hxm_create_module
_hxm_create_module.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_create_module.restype = _hxmdata
_hxm_create_module.errcheck = _hx_check_errors

_hxm_get_module = _lhx.hxm_get_module
_hxm_get_module.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_get_module.restype = _hxmdata
_hxm_get_module.errcheck = _hx_check_errors

_hxm_load_module_from_file = _lhx.hxm_load_module_from_file
_hxm_load_module_from_file.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_char_p]
_hxm_load_module_from_file.restype = _hxmdata
_hxm_load_module_from_file.errcheck = _hx_check_errors

_hxm_add_module_lookup_path = _lhx.hxm_add_module_lookup_path
_hxm_add_module_lookup_path.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_add_module_lookup_path.restype = None
_hxm_add_module_lookup_path.errcheck = _hx_check_errors

_hxm_clear_module_lookup_paths = _lhx.hxm_clear_module_lookup_paths
_hxm_clear_module_lookup_paths.argtypes = [_c.c_void_p]
_hxm_clear_module_lookup_paths.restype = None
_hxm_clear_module_lookup_paths.errcheck = _hx_check_errors

_hxm_module_get = _lhx.hxm_module_get
_hxm_module_get.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_module_get.restype = _hxmdata
_hxm_module_get.errcheck = _hx_check_errors

_hxm_module_set = _lhx.hxm_module_set
_hxm_module_set.argtypes = [_c.c_void_p, _c.c_char_p, _hxmdata]
_hxm_module_set.restype = None
_hxm_module_set.errcheck = _hx_check_errors

_hxm_module_run = _lhx.hxm_module_run
_hxm_module_run.argtypes = [_c.c_void_p, _c.c_void_p, _c.c_char_p, _c.c_int]
_hxm_module_run.restype = None
_hxm_module_run.errcheck = _hx_check_errors

_hxm_module_run_main = _lhx.hxm_module_run_main
_hxm_module_run_main.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_int]
_hxm_module_run_main.restype = None
_hxm_module_run_main.errcheck = _hx_check_errors


_hxm_create_map = _lhx.hxm_create_map
_hxm_create_map.argtypes = [_c.c_void_p]
_hxm_create_map.restype = _hxmdata
_hxm_create_map.errcheck = _hx_check_errors

_hxm_map_is_defined = _lhx.hxm_map_is_defined
_hxm_map_is_defined.argtypes = [_c.c_void_p, _hxmdata]
_hxm_map_is_defined.restype = _c.c_bool
_hxm_map_is_defined.errcheck = _hx_check_errors

_hxm_map_get = _lhx.hxm_map_get
_hxm_map_get.argtypes = [_c.c_void_p, _hxmdata]
_hxm_map_get.restype = _hxmdata
_hxm_map_get.errcheck = _hx_check_errors

_hxm_map_set = _lhx.hxm_map_set
_hxm_map_set.argtypes = [_c.c_void_p, _hxmdata, _hxmdata]
_hxm_map_set.restype = None
_hxm_map_set.errcheck = _hx_check_errors

_hxm_map_add = _lhx.hxm_map_add
_hxm_map_add.argtypes = [_c.c_void_p, _hxmdata]
_hxm_map_add.restype = None
_hxm_map_add.errcheck = _hx_check_errors

_hxm_map_clear = _lhx.hxm_map_clear
_hxm_map_clear.argtypes = [_c.c_void_p]
_hxm_map_clear.restype = None
_hxm_map_clear.errcheck = _hx_check_errors

_hxm_map_count = _lhx.hxm_map_count
_hxm_map_count.argtypes = [_c.c_void_p]
_hxm_map_count.restype = _c.c_longlong
_hxm_map_count.errcheck = _hx_check_errors


_hxm_create_expr = _lhx.hxm_create_expr
_hxm_create_expr.argtypes = [_c.c_void_p, _c.c_int]
_hxm_create_expr.restype = _hxmdata
_hxm_create_expr.errcheck = _hx_check_errors

_hxm_expr_index = _lhx.hxm_expr_index
_hxm_expr_index.argtypes = [_c.c_void_p]
_hxm_expr_index.restype = _c.c_int
_hxm_expr_index.errcheck = _hx_check_errors


_hxm_create_string = _lhx.hxm_create_string
_hxm_create_string.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_create_string.restype = _hxmdata
_hxm_create_string.errcheck = _hx_check_errors

_hxm_create_string_2 = _lhx.hxm_create_string
_hxm_create_string_2.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_int]
_hxm_create_string_2.restype = _hxmdata
_hxm_create_string_2.errcheck = _hx_check_errors

_hxm_string = _lhx.hxm_string
_hxm_string.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_int]
_hxm_string.restype = _c.c_int
_hxm_string.errcheck = _hx_check_errors

_hxm_function_type_2 = _c.CFUNCTYPE(
    None,
    _c.c_void_p,
    _c.POINTER(_hxmdata),
    _c.c_int,
    _c.POINTER(_hxmdata),
    _c.c_void_p
)

_hxm_create_function_2 = _lhx.hxm_create_function_2
_hxm_create_function_2.argtypes = [_c.c_void_p, _c.c_char_p, _hxm_function_type_2, _c.c_void_p]
_hxm_create_function_2.restype = _hxmdata
_hxm_create_function_2.errcheck = _hx_check_errors

_hxm_function_call = _lhx.hxm_function_call
_hxm_function_call.argtypes = [_c.c_void_p, _c.POINTER(_hxmdata), _c.c_int]
_hxm_function_call.restype = _hxmdata
_hxm_function_call.errcheck = _hx_check_errors

_hxm_function_name = _lhx.hxm_function_name
_hxm_function_name.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_int]
_hxm_function_name.restype = _c.c_int
_hxm_function_name.errcheck = _hx_check_errors


_hxm_create_iterator = _lhx.hxm_create_iterator
_hxm_create_iterator.argtypes = [_c.c_void_p]
_hxm_create_iterator.restype = _hxmdata
_hxm_create_iterator.errcheck = _hx_check_errors

_hxm_iterator_next = _lhx.hxm_iterator_next
_hxm_iterator_next.argtypes = [_c.c_void_p, _c.POINTER(_hxmdata), _c.POINTER(_hxmdata)]
_hxm_iterator_next.restype = _c.c_bool
_hxm_iterator_next.errcheck = _hx_check_errors


_hxm_inc_ref = _lhx.hxm_inc_ref
_hxm_inc_ref.argtypes = [_c.c_void_p]
_hxm_inc_ref.restype = None
_hxm_inc_ref.errcheck = _hx_check_errors

_hxm_dec_ref = _lhx.hxm_dec_ref
_hxm_dec_ref.argtypes = [_c.c_void_p]
_hxm_dec_ref.restype = None
_hxm_dec_ref.errcheck = _hx_check_errors

_hxm_type = _lhx.hxm_type
_hxm_type.argtypes = [_c.c_size_t]
_hxm_type.restype = _c.c_int
_hxm_type.errcheck = _hx_check_errors


_hxm_check_type = _lhx.hxm_check_type
_hxm_check_type.argtypes = [_c.c_size_t, _c.c_int]
_hxm_check_type.restype = _c.c_bool
_hxm_check_type.errcheck = _hx_check_errors


_hxm_optimizer = _lhx.hxm_optimizer
_hxm_optimizer.argtypes = [_c.c_void_p]
_hxm_optimizer.restype = _c.c_void_p
_hxm_optimizer.errcheck = _hx_check_errors

_hxm_create_optimizer = _lhx.hxm_create_optimizer
_hxm_create_optimizer.argtypes = [_c.c_void_p]
_hxm_create_optimizer.restype = _hxmdata
_hxm_create_optimizer.errcheck = _hx_check_errors

_hxm_optimizer_handle = _lhx.hxm_optimizer_handle
_hxm_optimizer_handle.argtypes = [_c.c_void_p]
_hxm_optimizer_handle.restype = _hxmdata
_hxm_optimizer_handle.errcheck = _hx_check_errors

_hxm_optimizer_reset = _lhx.hxm_optimizer_reset
_hxm_optimizer_reset.argtypes = [_c.c_void_p]
_hxm_optimizer_reset.restype = None
_hxm_optimizer_reset.errcheck = _hx_check_errors

_hxm_create_expr = _lhx.hxm_create_expr
_hxm_create_expr.argtypes = [_c.c_void_p, _c.c_int]
_hxm_create_expr.restype = _hxmdata
_hxm_create_expr.errcheck = _hx_check_errors

_hxm_expr_handle = _lhx.hxm_expr_handle
_hxm_expr_handle.argtypes = [_c.c_void_p]
_hxm_expr_handle.restype = _hxmdata
_hxm_expr_handle.errcheck = _hx_check_errors

_hxm_expr_index = _lhx.hxm_expr_index
_hxm_expr_index.argtypes = [_c.c_void_p]
_hxm_expr_index.restype = _c.c_int
_hxm_expr_index.errcheck = _hx_check_errors

_hxm_handle_optimizer = _lhx.hxm_handle_optimizer
_hxm_handle_optimizer.argtypes = [_c.c_void_p]
_hxm_handle_optimizer.restype = _c.c_void_p
_hxm_handle_optimizer.errcheck = _hx_check_errors

_hxm_get_class = _lhx.hxm_get_class
_hxm_get_class.argtypes = [_hxmdata]
_hxm_get_class.restype = _hxmdata
_hxm_get_class.errcheck = _hx_check_errors

_hxm_class_name = _lhx.hxm_class_name
_hxm_class_name.argtypes = [_c.c_void_p, _c.c_char_p, _c.c_int]
_hxm_class_name.restype = _c.c_int
_hxm_class_name.errcheck = _hx_check_errors

_hxm_class_is_final = _lhx.hxm_class_is_final
_hxm_class_is_final.argtypes = [_c.c_void_p]
_hxm_class_is_final.restype = _c.c_bool
_hxm_class_is_final.errcheck = _hx_check_errors

_hxm_class_has_super_class = _lhx.hxm_class_has_super_class
_hxm_class_has_super_class.argtypes = [_c.c_void_p]
_hxm_class_has_super_class.restype = _c.c_bool
_hxm_class_has_super_class.errcheck = _hx_check_errors

_hxm_class_super_class = _lhx.hxm_class_super_class
_hxm_class_super_class.argtypes = [_c.c_void_p]
_hxm_class_super_class.restype = _hxmdata
_hxm_class_super_class.errcheck = _hx_check_errors

_hxm_class_is_subclass_of = _lhx.hxm_class_is_subclass_of
_hxm_class_is_subclass_of.argtypes = [_c.c_void_p, _c.c_void_p]
_hxm_class_is_subclass_of.restype = _c.c_bool
_hxm_class_is_subclass_of.errcheck = _hx_check_errors

_hxm_class_is_instance_of = _lhx.hxm_class_is_instance_of
_hxm_class_is_instance_of.argtypes = [_c.c_void_p, _hxmdata]
_hxm_class_is_instance_of.restype = _c.c_bool
_hxm_class_is_instance_of.errcheck = _hx_check_errors

_hxm_check_class_instance = _lhx.hxm_check_class_instance
_hxm_check_class_instance.argtypes = [_c.c_void_p, _hxmdata]
_hxm_check_class_instance.restype = None
_hxm_check_class_instance.errcheck = _hx_check_errors

_hxm_class_new_instance = _lhx.hxm_class_new_instance
_hxm_class_new_instance.argtypes = [_c.c_void_p, _c.POINTER(_hxmdata), _c.c_int]
_hxm_class_new_instance.restype = _hxmdata
_hxm_class_new_instance.errcheck = _hx_check_errors

_hxm_class_nb_members = _lhx.hxm_class_nb_members
_hxm_class_nb_members.argtypes = [_c.c_void_p]
_hxm_class_nb_members.restype = _c.c_int
_hxm_class_nb_members.errcheck = _hx_check_errors

_hxm_class_member_type = _lhx.hxm_class_member_type
_hxm_class_member_type.argtypes = [_c.c_void_p, _c.c_int]
_hxm_class_member_type.restype = _c.c_int
_hxm_class_member_type.errcheck = _hx_check_errors

_hxm_class_member_name = _lhx.hxm_class_member_name
_hxm_class_member_name.argtypes = [_c.c_void_p, _c.c_int, _c.c_char_p, _c.c_int]
_hxm_class_member_name.restype = _c.c_int
_hxm_class_member_name.errcheck = _hx_check_errors

_hxm_class_find_member = _lhx.hxm_class_find_member
_hxm_class_find_member.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_class_find_member.restype = _c.c_int
_hxm_class_find_member.errcheck = _hx_check_errors

_hxm_class_member_id = _lhx.hxm_class_member_id
_hxm_class_member_id.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_class_member_id.restype = _c.c_int
_hxm_class_member_id.errcheck = _hx_check_errors

_hxm_class_member_method = _lhx.hxm_class_member_method
_hxm_class_member_method.argtypes = [_c.c_void_p, _c.c_int]
_hxm_class_member_method.restype = _hxmdata
_hxm_class_member_method.errcheck = _hx_check_errors

_hxm_class_member_slot = _lhx.hxm_class_member_slot
_hxm_class_member_slot.argtypes = [_c.c_void_p, _c.c_int]
_hxm_class_member_slot.restype = _c.c_int
_hxm_class_member_slot.errcheck = _hx_check_errors

_hxm_class_member_get_property = _lhx.hxm_class_member_get_property
_hxm_class_member_get_property.argtypes = [_c.c_void_p, _c.c_int, _c.c_void_p]
_hxm_class_member_get_property.restype = _hxmdata
_hxm_class_member_get_property.errcheck = _hx_check_errors

_hxm_class_member_set_property = _lhx.hxm_class_member_set_property
_hxm_class_member_set_property.argtypes = [_c.c_void_p, _c.c_int, _c.c_void_p, _hxmdata]
_hxm_class_member_set_property.restype = None
_hxm_class_member_set_property.errcheck = _hx_check_errors

_hxm_class_member_is_readonly_property = _lhx.hxm_class_member_is_readonly_property
_hxm_class_member_is_readonly_property.argtypes = [_c.c_void_p, _c.c_int]
_hxm_class_member_is_readonly_property.restype = _c.c_bool
_hxm_class_member_is_readonly_property.errcheck = _hx_check_errors

_hxm_check_class_property = _lhx.hxm_check_class_property
_hxm_check_class_property.argtypes = [_c.c_void_p, _c.c_int]
_hxm_check_class_property.restype = None
_hxm_check_class_property.errcheck = _hx_check_errors

_hxm_class_nb_static_members = _lhx.hxm_class_nb_static_members
_hxm_class_nb_static_members.argtypes = [_c.c_void_p]
_hxm_class_nb_static_members.restype = _c.c_int
_hxm_class_nb_static_members.errcheck = _hx_check_errors

_hxm_class_static_member_name = _lhx.hxm_class_static_member_name
_hxm_class_static_member_name.argtypes = [_c.c_void_p, _c.c_int, _c.c_char_p, _c.c_int]
_hxm_class_static_member_name.restype = _c.c_int
_hxm_class_static_member_name.errcheck = _hx_check_errors

_hxm_class_find_static_member = _lhx.hxm_class_find_static_member
_hxm_class_find_static_member.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_class_find_static_member.restype = _c.c_int
_hxm_class_find_static_member.errcheck = _hx_check_errors

_hxm_class_static_member_id = _lhx.hxm_class_static_member_id
_hxm_class_static_member_id.argtypes = [_c.c_void_p, _c.c_char_p]
_hxm_class_static_member_id.restype = _c.c_int
_hxm_class_static_member_id.errcheck = _hx_check_errors

_hxm_class_get_static_member = _lhx.hxm_class_get_static_member
_hxm_class_get_static_member.argtypes = [_c.c_void_p, _c.c_int]
_hxm_class_get_static_member.restype = _hxmdata
_hxm_class_get_static_member.errcheck = _hx_check_errors

_hxm_class_set_static_member = _lhx.hxm_class_set_static_member
_hxm_class_set_static_member.argtypes = [_c.c_void_p, _c.c_int, _hxmdata]
_hxm_class_set_static_member.restype = None
_hxm_class_set_static_member.errcheck = _hx_check_errors

_hxm_ref_get_slot = _lhx.hxm_ref_get_slot
_hxm_ref_get_slot.argtypes = [_c.c_void_p, _c.c_int]
_hxm_ref_get_slot.restype = _hxmdata
_hxm_ref_get_slot.errcheck = _hx_check_errors

_hxm_ref_set_slot = _lhx.hxm_ref_set_slot
_hxm_ref_set_slot.argtypes = [_c.c_void_p, _c.c_int, _hxmdata]
_hxm_ref_set_slot.restype = None
_hxm_ref_set_slot.errcheck = _hx_check_errors

_HXMMT_CONSTRUCTOR = 0
_HXMMT_METHOD = 1
_HXMMT_PROPERTY = 2
_HXMMT_FIELD = 3


########



class HxmValue(object):
    __slots__ = "_modeler", "_data", "_ref"

    def __init__(self, modeler, data):
        self._modeler = modeler
        self._data = data
        self._ref = data.val.ref
        _hxm_inc_ref(self._ref)
        modeler._reference_scopes[-1]._references.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        _check_modeler(self._modeler)
        self.__del__()

    def __del__(self):
        if self._ref != 0 and self._modeler._modeler_ptr is not None:
            _hxm_dec_ref(self._ref)
            self._data = 0
            self._ref = 0

    def get_class(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return HxmClass(self._modeler, _hxm_get_class(self._data))

    def __eq__(self, other):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return (isinstance(other, HxmValue)
                and self._ref == other._ref)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return self._ref

########


def _create_hxm_data(modeler, value):
    if value is None:
        return _hxmdata()
    elif isinstance(value, bool):
        result = _hxmdata()
        result.val.intValue = value
        result.type = 5
        return result
    elif isinstance(value, int):
        result = _hxmdata()
        result.val.intValue = value
        result.type = 1
        return result
    elif isinstance(value, float):
        result = _hxmdata()
        result.val.dblValue = value
        result.type = 2
        return result
    elif isinstance(value, HxExpression):
        _check_attached_optimizer(modeler, value._optimizer)
        return _hxm_create_expr(value._optimizer._handle_modeler_ptr, value.index)
    elif _is_string(value):
        raw_str = _encode_string(value)
        return _hxm_create_string_2(modeler._modeler_ptr, raw_str, len(raw_str))
    elif (isinstance(value, HxmValue)):
        return value._data
    elif isinstance(value, numbers.Integral):
        result = _hxmdata()
        result.val.intValue = int(value)
        result.type = 1
        return result
    elif isinstance(value, numbers.Real):
        result = _hxmdata()
        result.val.dblValue = float(value)
        result.type = 2
        return result
    else:
        raise TypeError("Cannot convert value of type '{}' to a valid modeler type."
                        .format(type(value)))

def _extract_hxm_data(modeler, data):
    if data.type == 0:
        return None
    elif data.type == 1:
        return data.val.intValue
    elif data.type == 2:
        return data.val.dblValue
    elif data.type == 5:
        if data.val.intValue == 1: return True
        else: return False

    value_type = _hxm_type(data.type)
    if value_type == 3:
        _hxm_inc_ref(data.ref)
        try:
            res = _read_string(lambda buf, x: _hxm_string(data.ref, buf, x))
            return res
        finally:
            _hxm_dec_ref(data.ref)
    elif value_type == 4:
        _hxm_inc_ref(data.ref)
        try:
            handle = _hxm_expr_handle(data.ref)
            expr_id = _hxm_expr_index(data.ref)
            optimizer = _retrieve_optimizer_from_handle(modeler, handle.ref)
            return HxExpression(optimizer, expr_id)
        finally:
            _hxm_dec_ref(data.ref)
    elif value_type == 5:
        return HxmFunction(modeler, data)
    elif value_type == 6:
        return HxmMap(modeler, data)
    elif value_type == 7:
        return HxmModule(modeler, data)
    elif value_type == 8:
        return HxmClass(modeler, data)
    else:
        return HxmValue(modeler, data)


def _check_modeler(modeler):
    if modeler._modeler_ptr is None:
        raise HxError(HxErrorCode.API, "Cannot perform the asked operation on a deleted environment.",
                "hxmutils.py", "_check_modeler", -1)

def _check_attached_optimizer(modeler, optimizer):
    if optimizer._handle_modeler_ptr is None or optimizer not in modeler._attached_optimizers:
        raise HxError(HxErrorCode.API,
                "The optimizer is not associated to a modeler. Create your optimizer with the method LSPModeler.create_optimizer() instead",
                "hxmutils.py", "_check_attached_optimizer", -1)

def _check_ref(reference):
    if reference == 0:
        raise HxError(HxErrorCode.API, "Cannot call this method on a disposed object.",
                "hxmutils.py", "_check_ref", -1)


def _retrieve_optimizer_from_handle(modeler, handle):
    for optimizer in modeler._attached_optimizers:
        if optimizer._handle_modeler_ptr == handle:
            return optimizer
    
    raise HxError(HxErrorCode.API,
            "Unable to retrieve HxExpressions based on a HexalyOptimizer model that was not created in the Python API.",
            "hxmutils.py", "_retrieve_optimizer_from_handle", -1)


########


class HxmModule(HxmValue):
    __slots__ = ()

    def __init__(self, modeler, data):
        super(HxmModule, self).__init__(modeler, data)

    def get_name(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _read_string(lambda buf, x: _hxm_module_name(self._ref, buf, x))

    def run(self, optimizer, *args):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        _check_attached_optimizer(self._modeler, optimizer)
        # Required since Jupyter overwrites the signal handler previously set without chaining it.
        _hx_define_signal_handler("SIGINT")
        str_args = _encode_strings_no_null(args)
        _hxm_module_run(self._ref, optimizer._optimizer_modeler_ptr, str_args, len(str_args))
        
    def run_main(self, *args):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        # Required since Jupyter overwrites the signal handler previously set without chaining it.
        _hx_define_signal_handler("SIGINT")
        str_args = _encode_strings_no_null(args)
        _hxm_module_run_main(self._ref, str_args, len(str_args))
        
    def __getitem__(self, var_name):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        result = _hxm_module_get(self._ref, _encode_string_no_null(var_name))
        return _extract_hxm_data(self._modeler, result)

    def __setitem__(self, var_name, value):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_val = _create_hxm_data(self._modeler, value)
        _hxm_module_set(self._ref, _encode_string_no_null(var_name), hxm_val)

    def __delitem__(self, var_name):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        _hxm_module_set(self._ref, _encode_string_no_null(var_name), _hxmdata())

    def __contains__(self, key):
        return self.__getitem__(key) is not None

    name = property(get_name)

########


class HxmMap(HxmValue):
    __slots__ = ()

    def __init__(self, modeler, data):
        super(HxmMap, self).__init__(modeler, data)

    def __len__(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_map_count(self._ref)

    def __getitem__(self, key):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_key = _create_hxm_data(self._modeler, key)
        hxm_val = _hxm_map_get(self._ref, hxm_key)
        return _extract_hxm_data(self._modeler, hxm_val)

    def __setitem__(self, key, value):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_key = _create_hxm_data(self._modeler, key)
        hxm_val = _create_hxm_data(self._modeler, value)
        _hxm_map_set(self._ref, hxm_key, hxm_val)

    def __delitem__(self, key):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_key = _create_hxm_data(self._modeler, key)
        _hxm_map_set(self._ref, hxm_key, _hxmdata())

    def __contains__(self, key):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_key = _create_hxm_data(self._modeler, key)
        return _hxm_map_is_defined(self._ref, hxm_key)

    def __iter__(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        iterator_ref = _hxm_create_iterator(self._ref).val.ref
        with _hxm_ref_guard(iterator_ref):
            key_data = _hxmdata()
            value_data = _hxmdata()
            while _hxm_iterator_next(iterator_ref, _c.byref(key_data), _c.byref(value_data)):
                yield _extract_hxm_data(self._modeler, key_data), _extract_hxm_data(self._modeler, value_data)
                _check_modeler(self._modeler)
                _check_ref(self._ref)
                key_data = _hxmdata()
                value_data = _hxmdata()

    def clear(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_map_clear(self._ref)

    def append(self, value):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_val = _create_hxm_data(self._modeler, value)
        _hxm_map_add(self._ref, hxm_val)

    def get(self, key, default = None):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_key = _create_hxm_data(self._modeler, key)
        hxm_val = _hxm_map_get(self._ref, hxm_key)
        if hxm_val.type == 0: return default
        return _extract_hxm_data(self._modeler, hxm_val)




########



class HxmField(object):
    __slots__ = "_modeler", "_class_ptr", "_slot_id"

    def __init__(self, modeler, class_ptr, slot_id):
        self._modeler = modeler
        self._class_ptr = class_ptr
        self._slot_id = slot_id
        _hxm_inc_ref(self._class_ptr)
        modeler._reference_scopes[-1]._references.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        _check_modeler(self._modeler)
        self.__del__()

    def __del__(self):
        if self._class_ptr != 0 and self._modeler._modeler_ptr is not None:
            _hxm_dec_ref(self._class_ptr)
            self._class_ptr = 0

    def get(self, obj):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        hxm_val = _create_hxm_data(self._modeler, obj)
        _hxm_check_class_instance(self._class_ptr, hxm_val)
        result = _hxm_ref_get_slot(hxm_val.val.ref, self._slot_id)
        return _extract_hxm_data(self._modeler, result)

    def set(self, obj, value):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        hxm_obj = _create_hxm_data(self._modeler, obj)
        hxm_val = _create_hxm_data(self._modeler, value)
        _hxm_check_class_instance(self._class_ptr, hxm_obj)
        _hxm_ref_set_slot(hxm_obj.val.ref, self._slot_id, hxm_val)

    def __eq__(self, other):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        return (isinstance(other, HxmField)
                and self._class_ptr == other._class_ptr
                and self._slot_id == other._slot_id)

    def __ne__(self, other):
        return not self.__eq__(other)

########



class HxmProperty(object):
    __slots__ = "_modeler", "_class_ptr", "_member_id"

    def __init__(self, modeler, class_ptr, member_id):
        self._modeler = modeler
        self._class_ptr = class_ptr
        self._member_id = member_id
        _hxm_inc_ref(self._class_ptr)
        modeler._reference_scopes[-1]._references.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        _check_modeler(self._modeler)
        self.__del__()

    def __del__(self):
        if self._class_ptr != 0 and self._modeler._modeler_ptr is not None:
            _hxm_dec_ref(self._class_ptr)
            self._class_ptr = 0

    def get(self, obj):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        lsp_val = _create_hxm_data(self._modeler, obj)
        _hxm_check_class_instance(self._class_ptr, lsp_val)
        result = _hxm_class_member_get_property(self._class_ptr, self._member_id, lsp_val.val.ref)
        return _extract_hxm_data(self._modeler, result)

    def set(self, obj, value):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        lsp_obj = _create_hxm_data(self._modeler, obj)
        lsp_val = _create_hxm_data(self._modeler, value)
        _hxm_check_class_instance(self._class_ptr, lsp_obj)
        _hxm_class_member_set_property(self._class_ptr, self._member_id, lspobj.val.ref, lsp_val)

    def is_readonly(self):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        return _hxm_class_member_is_readonly_property(self._class_ptr, self._member_id)

    def __eq__(self, other):
        _check_modeler(self._modeler)
        _check_ref(self._class_ptr)
        return (isinstance(other, LSPField)
                and self._class_ptr == other._class_ptr
                and self._member_id == other._member_id)

    def __ne__(self, other):
        return not self.__eq__(other)

    readonly = property(is_readonly)

########


class HxmReferenceScope(object):
    __slots__ = "_modeler", "_references"

    def __init__(self, modeler):
        self._modeler = modeler
        self._references = []
        modeler._reference_scopes.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        _check_modeler(self._modeler)
        self.__del__()

    def __del__(self):
        if self._modeler._modeler_ptr is None:
            return

        try:
            self._modeler._reference_scopes.remove(self)
        except ValueError:
            pass

        for ref in self._references:
            ref.__exit__()
        self._references = []



########




class HxmFunction(HxmValue):
    __slots__ = "_args_array",

    def __init__(self, modeler, data):
        super(HxmFunction, self).__init__(modeler, data)
        self._args_array = None

    def get_name(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _read_string(lambda buf, x: _hxm_function_name(self._ref, buf, x))

    def call(self, *args):
        return self._generic_call(None, *args)

    def call_this(self, this, *args):
        return self._generic_call(this, *args)

    def _generic_call(self, this, *args):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        if self._args_array is None or len(self._args_array) < len(args) + 1:
            self._args_array = (_hxmdata * (len(args) + 1))()

        self._args_array[0] = _create_hxm_data(self._modeler, this)
        for i, arg in enumerate(args, start=1):
            self._args_array[i] = _create_hxm_data(self._modeler, arg)

        result = _hxm_function_call(self._ref, self._args_array, len(args) + 1)
        return _extract_hxm_data(self._modeler, result)

    def __call__(self, *args):
        return self.call(*args)


    name = property(get_name)
########



class HxmClass(HxmValue):
    __slots__ = "_args_array"

    def __init__(self, modeler, data):
        super(HxmClass, self).__init__(modeler, data)
        self._args_array = None

    def __call__(self, *args):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        if (self._args_array is None or len(self._args_array) < len(args) + 1):
            self._args_array = (_hxmdata * (len(args) + 1))()
        
        for i, arg in enumerate(args, start=1):
            self._args_array[i] = _create_hxm_data(self._modeler, arg)
        
        result = _hxm_class_new_instance(self._ref, self._args_array, len(args) + 1)
        return _extract_hxm_data(self._modeler, result)

    def get_name(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _read_string(lambda buf, x: _hxm_class_name(self._ref, buf, x))

    def is_final(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_is_final(self._ref)

    def is_subclass_of(self, other):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_val = _create_hxm_data(self._modeler, other)
        return _hxm_class_is_subclass_of(self._ref, hxm_val.val.ref)

    def has_super_class(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_has_super_class(self._ref)

    def get_super_class(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        result = _hxm_class_super_class(self._ref)
        return _extract_hxm_data(self._modeler, result)

    def is_instance(self, value):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        hxm_val =  _create_hxm_data(self._modeler, value)
        return _hxm_class_is_instance_of(self._ref, hxm_val)

    def get_nb_members(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_nb_members(self._ref)

    def get_member_name(self, member_id):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _read_string(lambda buf, x: _hxm_class_member_name(self._ref, member_id, buf, x))

    def find_member_id(self, member_name):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_find_member(self._ref, _encode_string_no_null(member_name))

    def is_member(self, member_name):
        return self.find_member_id(member_name) != -1

    def is_method(self, member_name_or_id):
        return self._is_member_type(member_name_or_id, _HXMMT_METHOD)

    def is_field(self, member_name_or_id):
        return self._is_member_type(member_name_or_id, _HXMMT_FIELD)

    def is_property(self, member_name_or_id):
        return self._is_member_type(member_name_or_id, _HXMMT_PROPERTY)

    def get_member(self, member_name_or_id):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        member_id = self._extract_member_id(member_name_or_id)
        member_type = _hxm_class_member_type(self._ref, member_id)
        if member_type == _HXMMT_FIELD:
            slot_id = _hxm_class_member_slot(self._ref, member_id)
            return HxmField(self._modeler, self._ref, slot_id)
        elif member_type == _HXMMT_PROPERTY:
            return HxmProperty(self._modeler, self._ref, member_id)
        elif member_type == _HXMMT_METHOD:
            hxm_val = _hxm_class_member_method(self._ref, member_id)
            return HxmFunction(self._modeler, hxm_val)
        else:
            raise HxError(HxErrorCode.API, 
                    "The constructor of a class cannot be directly obtained. " +
                    "To instantiate an object of this class use the method HxmClass.__call__() instead.", 
                    "hxmclass.py", "get_member", -1)

    def get_nb_static_members(self):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_nb_static_members(self._ref)

    def get_static_member_name(self, static_member_id):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _read_string(lambda buf, x: _hxm_class_static_member_name(self._ref, static_member_id, buf, x))

    def find_static_member_id(self, static_member_name):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_find_static_member(self._ref, _encode_string_no_null(static_member_name))

    def is_static_member(self, static_member_name):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        return _hxm_class_find_static_member(self._ref, _encode_string_no_null(static_member_name)) != -1

    def get_static_member(self, static_member_name_or_id):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        static_member_id = self._extract_static_member_id(static_member_name_or_id)
        result = _hxm_class_get_static_member(self._ref, static_member_id)
        return _extract_hxm_data(self._modeler, result)

    def set_static_member(self, static_member_name_or_id, value):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        static_member_id = self._extract_static_member_id(static_member_name_or_id)
        hxm_val = _create_hxm_data(self._modeler, value)
        _hxm_class_set_static_member(self._ref, static_member_id, hxm_val)

    def _extract_member_id(self, member_name_or_id):
        if isinstance(member_name_or_id, int):
            return member_name_or_id
        return _hxm_class_member_id(self._ref, _encode_string_no_null(member_name_or_id))

    def _extract_static_member_id(self, static_member_name_or_id):
        if isinstance(static_member_name_or_id, int):
            return static_member_name_or_id
        return _hxm_class_find_static_member(self._ref, _encode_string_no_null(static_member_name_or_id))

    def _is_member_type(self, member_name_or_id, type):
        _check_modeler(self._modeler)
        _check_ref(self._ref)
        member_id = (member_name_or_id if isinstance(member_name_or_id, int) 
            else _hxm_class_find_member(self._ref, _encode_string_no_null(member_name_or_id)))
        if member_id == -1:
            return False
        return _hxm_class_member_type(self._ref, member_id) == type

    name = property(get_name)
    final = property(is_final)
    super_class = property(get_super_class)
    nb_members = property(get_nb_members)
    nb_static_members = property(get_nb_static_members)
########



class HexalyModeler(object):
    __slots__ = "_modeler_ptr", "_native_functions", "_attached_optimizers", "_reference_scopes", "_stdout", "_stderr", "_stdout_funcs", "_stderr_funcs"

    def __init__(self):
        self._modeler_ptr = _hxm_create_modeler()
        self._native_functions = []
        self._attached_optimizers = []
        self._reference_scopes = []
        HxmReferenceScope(self)
        self.set_stdout(sys.stdout)
        self.set_stderr(sys.stderr)

    def delete(self):
        if self._modeler_ptr is None:
            return

        for i in reversed(range(len(self._reference_scopes))):
            self._reference_scopes[i].__del__()

        _hxm_delete_modeler(self._modeler_ptr)
        self._modeler_ptr = None

    def __del__(self):
        self.delete()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.delete()

    def create_optimizer(self):
        _check_modeler(self)
        self._attached_optimizers = [optimizer for optimizer in self._attached_optimizers if optimizer._optimizer_modeler_ptr is not None]
        
        optimizer_modeler = _hxm_create_optimizer(self._modeler_ptr).val.ref
        handle_ref = _hxm_optimizer_handle(optimizer_modeler).val.ref
        optimizer_ref = _hxm_handle_optimizer(handle_ref)

        optimizer = HexalyOptimizer(optimizer_ref, handle_ref, optimizer_modeler)
        self._attached_optimizers.append(optimizer)
        self._reference_scopes[-1]._references.append(optimizer)
        return optimizer

    def get_module(self, module_name):
        _check_modeler(self)
        data = _hxm_get_module(self._modeler_ptr, _encode_string_no_null(module_name))
        return HxmModule(self, data)

    def load_module(self, module_name, filepath):
        _check_modeler(self)
        data = _hxm_load_module_from_file(self._modeler_ptr, _encode_string_no_null(module_name), _encode_string_no_null(filepath))
        return HxmModule(self, data)

    def create_module(self, module_name):
        _check_modeler(self)
        data = _hxm_create_module(self._modeler_ptr, _encode_string_no_null(module_name))
        return HxmModule(self, data)
    
    def add_module_lookup_path(self, path):
        _check_modeler(self)
        _hxm_add_module_lookup_path(self._modeler_ptr, _encode_string_no_null(path))

    def clear_module_lookup_paths(self):
        _check_modeler(self)
        _hxm_clear_module_lookup_paths(self._modeler_ptr)

    def create_map(self):
        _check_modeler(self)
        return HxmMap(self, _hxm_create_map(self._modeler_ptr))

    def create_function(self, name, func=None):
        _check_modeler(self)
        if func is None and name is not None:
                func, name = name, func
        arguments = []

        def native_function(_1, args, nb_args, result, _2):
            del arguments[:]
            try:
                for i in range(1, nb_args):
                    arguments.append(_extract_hxm_data(self, args[i]))
                result[0] = _create_hxm_data(self, func(self, *arguments))
            except:
                _hx_set_pending_error(sys.exc_info()[1])
                _hx_interrupt(_encode_string(repr(sys.exc_info()[1])), None)
        
        native_func = _hxm_function_type_2(native_function)
        self._native_functions.append(native_func)
        data = _hxm_create_function_2(self._modeler_ptr, _encode_string_no_null(name), native_func, None)
        return HxmFunction(self, data)

    def get_stdout(self):
        return self._stdout

    def set_stdout(self, stream):
        self._stdout_funcs = self._set_std_stream(1, stream)
        self._stdout = stream

    def get_stderr(self):
        return self._stderr

    def set_stderr(self, stream):
        self._stderr_funcs = self._set_std_stream(2, stream)
        self._stderr = stream

    def _set_std_stream(self, fd, stream):
        def writer_function(_1, content, length, _2):
            try:
                if stream is None:
                    return
                stream.write(_decode_subset_bytes(content, length))
            except:
                _hx_set_pending_error(sys.exc_info()[1])
                _hx_interrupt(_encode_string(repr(sys.exc_info()[1])), None)

        def flusher_function(_1, _2):
            try:
                if stream is None:
                    return
                stream.flush()
            except:
                _hx_set_pending_error(sys.exc_info()[1])
                _hx_interrupt(_encode_string(repr(sys.exc_info()[1])), None)

        writer_func = _hxm_stream_writer_type(writer_function)
        flusher_func = _hxm_stream_flusher_type(flusher_function)
        _hxm_set_std_stream(self._modeler_ptr, fd, writer_func, flusher_func, None)
        return (writer_func, flusher_func)

    def __eq__(self, other):
        return (isinstance(other, HexalyModeler)
                and self._modeler_ptr == other._modeler_ptr)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self._modeler_ptr

    stdout = property(get_stdout, set_stdout)
    stderr = property(get_stderr, set_stderr)
########

