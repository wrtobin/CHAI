#ifndef MANAGED_PTR_H_
#define MANAGED_PTR_H_

#include "chai/ChaiMacros.hpp"

// Standard libary headers
#include <cstddef>
#include <tuple>

namespace chai {
   template <typename T>
   class managed_ptr;

#ifdef __CUDACC__
   namespace detail {
      ///
      /// @author Alan Dayton
      ///
      /// Creates a new T on the device.
      ///
      /// @param[out] devicePtr Used to return the device pointer to the new T
      /// @param[in]  args The arguments to T's constructor
      ///
      /// @note Cannot capture argument packs in an extended device lambda,
      ///       so explicit kernel is needed.
      ///
      template <typename T,
                typename... Args>
      __global__ void make_on_device(T*& devicePtr, Args... args)
      {
         devicePtr = new T(std::forward<Args>(args)...);
      }

      ///
      /// @author Alan Dayton
      ///
      /// Creates a new object on the device by calling the given factory method.
      ///
      /// @param[out] devicePtr Used to return the device pointer to the new object
      /// @param[in]  f The factory method (must be a __device__ or __host__ __device__
      ///                method
      /// @param[in]  args The arguments to the factory method
      ///
      /// @note Cannot capture argument packs in an extended device lambda,
      ///       so explicit kernel is needed.
      ///
      template <typename T,
                typename F,
                typename... Args>
      __global__ void make_on_device_from_factory(T*& devicePtr, F f, Args... args)
      {
         devicePtr = f(std::forward<Args>(args)...);
      }

      ///
      /// @author Alan Dayton
      ///
      /// Destroys the device pointer.
      ///
      /// @param[out] devicePtr The device pointer to call delete on
      ///
      template <typename T>
      __global__ void destroy_on_device(T*& devicePtr)
      {
         if (devicePtr) {
            delete devicePtr;
         }
      }

      ///
      /// @author Alan Dayton
      ///
      /// Gets the device pointer from the managed_ptr.
      ///
      /// @param[out] devicePtr Used to return the device pointer
      /// @param[in]  other The managed_ptr from which to extract the device pointer
      ///
      template <typename T>
      __global__ void get_on_device(T*& devicePtr,
                                    const managed_ptr<T>& other)
      {
         devicePtr = other.get(GPU);
      }

      ///
      /// @author Alan Dayton
      ///
      /// Converts the underlying pointer on the device using static_cast.
      ///
      /// @param[out] devicePtr The device pointer that will contain the result of
      ///                           calling static_cast on the pointer contained by
      ///                           the given managed_ptr
      /// @param[in] other The managed_ptr to share ownership with and whose pointer to
      ///                      convert using static_cast
      ///
      template <typename T, typename U>
      __global__ void static_pointer_cast_on_device(T*& devicePtr,
                                                    const managed_ptr<U>& other)
      {
         devicePtr = static_cast<T*>(other.get(GPU));
      }

      ///
      /// @author Alan Dayton
      ///
      /// Converts the underlying pointer on the device using const_cast.
      ///
      /// @param[out] devicePtr The device pointer that will contain the result of
      ///                           calling const_cast on the pointer contained by
      ///                           the given managed_ptr
      /// @param[in] other The managed_ptr to share ownership with and whose pointer to
      ///                      convert using const_cast
      ///
      template <typename T, typename U>
      __global__ void const_pointer_cast_on_device(T*& devicePtr,
                                                   const managed_ptr<U>& other) {
         devicePtr = const_cast<T*>(other.get(GPU));
      }

      ///
      /// @author Alan Dayton
      ///
      /// Converts the underlying pointer on the device using reinterpret_cast.
      ///
      /// @param[out] devicePtr The device pointer that will contain the result of
      ///                           calling reinterpret_cast on the pointer contained by
      ///                           the given managed_ptr
      /// @param[in] other The managed_ptr to share ownership with and whose pointer to
      ///                      convert using reinterpret_cast
      ///
      template <typename T, typename U>
      __global__ void reinterpret_pointer_cast_on_device(T*& devicePtr,
                                                         const managed_ptr<U>& other) {
         devicePtr = reinterpret_cast<T*>(other.get(GPU));
      }

      ///
      /// @author Alan Dayton
      ///
      /// Creates a new T on the device.
      ///
      /// @param[in]  args The arguments to T's constructor
      ///
      /// @return The device pointer to the new T
      ///
      template <typename T,
                typename... Args>
      CHAI_HOST T* make_on_device(Args&&... args) {
         T* devicePtr;
         make_on_device<<<1, 1>>>(devicePtr, args...);
         cudaDeviceSynchronize();
         return devicePtr;
      }

      ///
      /// @author Alan Dayton
      ///
      /// Calls a factory method to create a new object on the device.
      ///
      /// @param[in]  f    The factory method
      /// @param[in]  args The arguments to the factory method
      ///
      /// @return The device pointer to the new object
      ///
      template <typename T,
                typename F,
                typename... Args>
      CHAI_HOST T* make_on_device_from_factory(F f, Args&&... args) {
         T* devicePtr;
         make_on_device_from_factory<T><<<1, 1>>>(devicePtr, f, args...);
         cudaDeviceSynchronize();
         return devicePtr;
      }

      ///
      /// @author Alan Dayton
      ///
      /// Gets the device pointer from the managed_ptr.
      ///
      /// @param[in] other The managed_ptr from which to extract the device pointer
      ///
      template <typename T>
      T* get_on_device(const managed_ptr<T>& other) {
         T* devicePtr;
         get_on_device<<<1, 1>>>(devicePtr, other);
         cudaDeviceSynchronize();
         return devicePtr;
      }

      ///
      /// @author Alan Dayton
      ///
      /// Converts the underlying pointer on the device using static_cast.
      ///
      /// @param[in] other The managed_ptr to share ownership with and whose pointer to
      ///                      convert using static_cast
      ///
      template <typename T, typename U>
      CHAI_HOST T* static_pointer_cast_on_device(const managed_ptr<U>& other) noexcept {
         T* devicePtr;
         static_pointer_cast_on_device<<<1, 1>>>(devicePtr, other);
         cudaDeviceSynchronize();
         return devicePtr;
      }

      /// @author Alan Dayton
      ///
      /// Converts the underlying pointer on the device using const_cast.
      ///
      /// @param[in] other The managed_ptr to share ownership with and whose pointer to
      ///                      convert using const_cast
      ///
      template <typename T, typename U>
      CHAI_HOST T* const_pointer_cast_on_device(const managed_ptr<U>& other) noexcept {
         T* devicePtr;
         const_pointer_cast_on_device<<<1, 1>>>(devicePtr, other);
         cudaDeviceSynchronize();
         return devicePtr;
      }

      ///
      /// @author Alan Dayton
      ///
      /// Converts the underlying pointer on the device using reinterpret_cast.
      ///
      /// @param[in] other The managed_ptr to share ownership with and whose pointer to
      ///                      convert using reinterpret_cast
      ///
      template <typename T, typename U>
      CHAI_HOST T* reinterpret_pointer_cast_on_device(const managed_ptr<U>& other) noexcept {
         T* devicePtr;
         reinterpret_pointer_cast_on_device<<<1, 1>>>(devicePtr, other);
         cudaDeviceSynchronize();
         return devicePtr;
      }
   }

   class managed_ptr_record {
      public:
         managed_ptr_record() = delete;

#if 0
         template <std::size_t N, typename U>
         managed_ptr_record(const ExecutionSpace(&spaces)[N],
                            const U*(&pointers)[N],
                            std::function<void(Action, ExecutionSpace, void*)> callback = [] (Action action,
    ExecutionSpace space,
    void* pointer) {
       switch (action) {
          case ACTION_FREE:
             switch (space) {
                case CPU:
                   delete static_cast<U*>(pointer);
                   break;
                case GPU:
                   detail::destroy_on_device<<<1, 1>>>(static_cast<U*>(pointer));
                   break;
             }
       }
    }) :
            m_numReferences(1),
            m_callback(callback)
         {
            for (int i = 0; i < N; ++i) {
               m_pointers[static_cast<size_t>(spaces[i])] = static_cast<void*>(pointers[i]);
            }
         }
#endif

         managed_ptr_record(std::function<void(Action, ExecutionSpace, void*&)> callback) : m_numReferences(1), m_callback(callback) {}

         size_t use_count() {
            return m_numReferences;
         }

         virtual void incrementReferenceCount() {
            m_numReferences++;
         }

         virtual void decrementReferenceCount() {
            m_numReferences--;

            if (m_numReferences == 0) {
               for (int space = NONE; space < NUM_EXECUTION_SPACES; ++space) {
                  m_callback(ACTION_FREE, static_cast<ExecutionSpace>(space), m_pointers[space]);
               }
            }
         }

         virtual void* getActivePointer() {
            ExecutionSpace activeSpace = getCurrentExecutionSpace();
            return get(activeSpace);
         }

         virtual void* get(ExecutionSpace space=NONE) {
            if (space == NONE) {
               space = ArrayManager::getInstance()->getDefaultAllocationSpace();
            }

            move(space);

            void* pointer = m_pointers[static_cast<size_t>(space)];

            if (!pointer) {
               m_callback(ACTION_ALLOC, space, pointer);
            }

            return pointer;
         }

         inline ExecutionSpace getCurrentExecutionSpace() {
            return ArrayManager::getInstance()->getExecutionSpace();
         }

      private:
         void* m_pointers[NUM_EXECUTION_SPACES]; /// The pointers
         size_t m_numReferences = 1; /// The reference counter
         ExecutionSpace m_lastSpace = NONE; /// The last space executed in
         std::function<void(Action, ExecutionSpace, void*&)> m_callback; /// Callback to handle events

         void move(ExecutionSpace space) {
            if (space != NONE && m_lastSpace != space) {
               m_lastSpace = space;
               m_callback(ACTION_MOVE, space, m_pointers[space]);
            }
         }
   };

   ///
   /// @class managed_ptr<T>
   /// @author Alan Dayton
   /// This wrapper calls new on both the GPU and CPU so that polymorphism can
   ///    be used on the GPU. It is modeled after std::shared_ptr, so it does
   ///    reference counting and automatically cleans up when the last reference
   ///    is destroyed. If we ever do multi-threading on the CPU, locking will
   ///    need to be added to the reference counter.
   /// Requirements:
   ///    The underlying type created (U in the first constructor) must be convertible
   ///       to T (e.g. T is a base class of U or there is a user defined conversion).
   ///    This wrapper does NOT automatically sync the GPU copy if the CPU copy is
   ///       updated and vice versa. The one exception to this is nested ManagedArrays
   ///       and managed_ptrs, but only if they are registered via the registerArguments
   ///       method. The factory methods make_managed and make_managed_from_factory
   ///       will register arguments passed to them automatically. Otherwise, if you
   ///       wish to keep the CPU and GPU instances in sync, you must explicitly modify
   ///       the object in both the CPU context and the GPU context.
   ///    Members of T that are raw pointers need to be initialized correctly with a
   ///       host or device pointer. If it is desired that these be kept in sync,
   ///       pass a ManagedArray to the make_managed or make_managed_from_factory
   ///       functions in place of a raw array. Or, if this is after the managed_ptr
   ///       has been constructed, use the same ManagedArray in both the CPU and GPU
   ///       contexts to initialize the raw pointer member and then register the
   ///       ManagedArray with the registerArguments method on the managed_ptr.
   ///       If only a raw array is passed to make_managed, accessing that member
   ///       will be valid only in the correct context. To prevent the accidental
   ///       use of that member in the wrong context, any methods that access raw
   ///       pointers not initialized in both contexts as previously described
   ///       should be __host__ only or __device__ only. Special care should be
   ///       taken when passing raw pointers as arguments to member functions.
   ///    Methods that can be called on the CPU and GPU must be declared with the
   ///       __host__ __device__ specifiers. This includes the constructors being
   ///       used and destructors.
   ///
   template <typename T>
   class managed_ptr {
      public:
         using element_type = T;

         ///
         /// @author Alan Dayton
         ///
         /// Default constructor.
         /// Initializes the reference count to 0.
         ///
         CHAI_HOST_DEVICE constexpr managed_ptr() noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Construct from nullptr.
         /// Initializes the reference count to 0.
         ///
         CHAI_HOST_DEVICE constexpr managed_ptr(std::nullptr_t) noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Constructs a managed_ptr from the given host and device pointers.
         ///
         /// @param[in] pointers The pointers to take ownership of
         ///
         managed_ptr(std::function<void(Action, ExecutionSpace, void*&)> callback,
                     ExecutionSpace activeSpace=NONE) :
            m_active_pointer(nullptr),
            m_pointer_record(new managed_ptr_record(callback))
         {
            m_active_pointer = static_cast<T*>(m_pointer_record->get(activeSpace));
         }

         ///
         /// @author Alan Dayton
         ///
         /// Constructs a managed_ptr from the given host and device pointers.
         ///
         /// @param[in] pointers The pointers to take ownership of
         ///
#if 0
         template <typename U>
         CHAI_HOST managed_ptr(std::initializer_list<ExecutionSpace>& spaces,
                               std::initializer_list<U*>& pointers,
                               std::function<void(Action, ExecutionSpace, void*)> callback,
                               ExecutionSpace activeSpace = NONE) :
#endif
#if 0
         template <std::size_t N, typename U>
         CHAI_HOST managed_ptr(const ExecutionSpace(&spaces)[N],
                               const U*(&pointers)[N],
                               std::function<void(Action, ExecutionSpace, void*)> callback,
                               ExecutionSpace activeSpace) :
            m_active_pointer(nullptr),
            m_pointer_record(new managed_ptr_record<T>(spaces, pointers, callback))
         {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

            m_active_pointer = static_cast<T*>(m_pointer_record->get(activeSpace));
         }
#endif

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         /// Constructs a copy of the given managed_ptr and increases the reference count.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr& other) noexcept :
            m_active_pointer(other.m_active_pointer),
            m_pointer_record(other.m_pointer_record)
         {
#ifndef __CUDA_ARCH__
            m_active_pointer = static_cast<T*>(m_pointer_record->get());
            incrementReferenceCount();
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Converting constructor.
         /// Constructs a copy of the given managed_ptr and increases the reference count.
         ///    U must be convertible to T.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         template <typename U>
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr<U>& other) noexcept :
            m_active_pointer(other.m_active_pointer),
            m_pointer_record(other.m_pointer_record)
         {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

#ifndef __CUDA_ARCH__
            m_active_pointer = static_cast<T*>(m_pointer_record->get());
            incrementReferenceCount();
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Aliasing constructor.
         /// Has the same ownership information as other, but holds a different pointer.
         ///
         /// @param[in] other The managed_ptr to copy ownership information from
         /// @param[in] hostPtr The host pointer to maintain a reference to
         /// @param[in] devicePtr The device pointer to maintain a reference to
         ///
         template <typename U>
         CHAI_HOST managed_ptr(const managed_ptr<U>& other,
                               std::function<void(Action, ExecutionSpace, void*&)> callback) noexcept :
            m_active_pointer(other.m_active_pointer),
            m_pointer_record(other.m_pointer_record)
         {
            m_active_pointer = static_cast<T*>(m_pointer_record->get());
            incrementReferenceCount();
         }

         ///
         /// @author Alan Dayton
         ///
         /// Destructor. Decreases the reference count and if this is the last reference,
         ///    clean up.
         ///
         CHAI_HOST_DEVICE ~managed_ptr() {
#ifndef __CUDA_ARCH__
            decrementReferenceCount();
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy assignment operator.
         /// Copies the given managed_ptr and increases the reference count.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr& other) noexcept {
            if (this != &other) {
#ifndef __CUDA_ARCH__
               decrementReferenceCount();
#endif

               m_active_pointer = other.m_active_pointer;
               m_pointer_record = other.m_pointer_record;

#ifndef __CUDA_ARCH__
               m_active_pointer = static_cast<T*>(m_pointer_record->get());
               incrementReferenceCount();
#endif
            }

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Conversion copy assignment operator.
         /// Copies the given managed_ptr and increases the reference count.
         ///    U must be convertible to T.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         template<class U>
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr<U>& other) noexcept {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

#ifndef __CUDA_ARCH__
            decrementReferenceCount();
#endif

            m_active_pointer = other.m_active_pointer;
            m_pointer_record = other.m_pointer_record;

#ifndef __CUDA_ARCH__
            m_active_pointer = static_cast<T*>(m_pointer_record->get());
            incrementReferenceCount();
#endif

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T* get() const { return m_active_pointer; }

         CHAI_HOST inline T* get(ExecutionSpace space) const {
            return static_cast<T*>(m_pointer_record->get(space));
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T* operator->() const { return m_active_pointer; }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU reference depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T& operator*() const { return *m_active_pointer; }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the number of managed_ptrs owning these pointers.
         ///
         CHAI_HOST std::size_t use_count() const {
            if (m_pointer_record) {
               return m_pointer_record->use_count();
            }
            else {
               return 0;
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns true if the contained pointer is not nullptr, false otherwise.
         ///
         CHAI_HOST_DEVICE inline explicit operator bool() const noexcept {
            return m_active_pointer != nullptr;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Implicit conversion operator to T*. Be careful when using this because
         /// data movement is not triggered by the raw pointer.
         ///
         CHAI_HOST_DEVICE inline operator T*() const {
#ifndef __CUDA_ARCH__
            //m_copier(m_copyArguments);
#endif
            return get();
         }

      private:
         T* m_cpu_pointer = nullptr;
         T* m_gpu_pointer = nullptr;
         managed_ptr_record* m_pointer_record = nullptr; /// The pointer record

         template <typename U>
         friend class managed_ptr; /// Needed for the converting constructor

         ///
         /// @author Alan Dayton
         ///
         /// Increments the reference count and calls the copy constructor to
         ///    trigger data movement.
         ///
         CHAI_HOST void incrementReferenceCount() {
            if (m_pointer_record) {
               m_pointer_record->incrementReferenceCount();
               m_active_pointer = static_cast<T*>(m_pointer_record->getActivePointer());
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Decrements the reference counter. If the resulting number of references
         ///    is 0, clean up the object.
         ///
         CHAI_HOST void decrementReferenceCount() {
            if (m_pointer_record) {
               m_pointer_record->decrementReferenceCount();

               if (m_pointer_record->use_count() == 0) {
                  delete m_pointer_record;
               }
            }
         }
   };

   template <typename, typename>
   struct typelist_concatenate;

   template <typename T, typename... Args>
   struct typelist_concatenate<T, std::tuple<Args...>> {
      using type = std::tuple<T, Args...>;
   };

   template <typename T, typename... Args>
   using typelist_concatenate_t = typename typelist_concatenate<T, Args...>::type;

   template <template <typename> class, typename...>
   struct filter;

   template <template <typename> class Predicate>
   struct filter<Predicate> {
      using type = std::tuple<>;
   };

   template <template <typename> class Predicate, typename T, typename... Args>
   struct filter<Predicate, T, Args...> {
      using type = typename std::conditional<Predicate<T>::value,
                                             typelist_concatenate_t<T, typename filter<Predicate, Args...>::type>,
                                             typename filter<Predicate, Args...>::type>::type;
   };

   template <template <typename> class Predicate, typename... Args>
   using filter_t = typename filter<Predicate, Args...>::type;

   template <typename T>
   std::tuple<managed_ptr<T>> getManagedArguments(managed_ptr<T> arg) {
      return std::forward_as_tuple(arg);
   }

   template <typename T>
   std::tuple<ManagedArray<T>> getManagedArguments(ManagedArray<T> arg) {
      return std::forward_as_tuple(arg);
   }

   template <typename T>
   std::tuple<> getManagedArguments(T arg) {
      return std::tuple<>();
   }

   template <typename>
   struct IsManaged : std::false_type {};

   template <typename T>
   struct IsManaged<ManagedArray<T>> : std::true_type {};

   template <typename T>
   struct IsManaged<managed_ptr<T>> : std::true_type {};

   template <typename T, typename... Args>
   filter_t<IsManaged, T, Args...> getManagedArguments(T arg, Args... args) {
      return std::tuple_cat(getManagedArguments(arg), getManagedArguments(args...));
   }

   // Taken from https://stackoverflow.com/questions/1198260/how-can-you-iterate-over-the-elements-of-an-stdtuple
   template <size_t ...I>
   struct index_sequence {};

   template <size_t N, size_t ...I>
   struct make_index_sequence : public make_index_sequence<N - 1, N - 1, I...> {};

   template <size_t ...I>
   struct make_index_sequence<0, I...> : public index_sequence<I...> {};

   template <typename T>
   void doFreeManagedArrays(T arg) {}

   template <typename T>
   void doFreeManagedArrays(ManagedArray<T> arg) {
      if (arg) {
         arg.free();
      }
   }

   // Adapted from https://stackoverflow.com/questions/1198260/how-can-you-iterate-over-the-elements-of-an-stdtuple
   template<typename ...T, size_t ...I>
   void freeManagedArraysHelper(std::tuple<T...> &ts, index_sequence<I...>) {
      //std::tie((doFreeManagedArrays(std::get<I>(ts)), 1) ... );
      doFreeManagedArrays(std::get<I>(ts)...);
   }

   template <typename... Args>
   void freeManagedArrays(std::tuple<Args...>& args) {
      return freeManagedArraysHelper(args, make_index_sequence<sizeof...(Args)>());
   }

   template <>
   void freeManagedArrays(std::tuple<>& args) {
      return;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @param[in] args The arguments to T's constructor
   ///
   template <typename T,
             typename... Args,
             typename std::enable_if<std::is_constructible<T, Args...>::value, int>::type = 0>
   CHAI_HOST managed_ptr<T> make_managed(Args&&... args) {
      static_assert(std::is_constructible<T, Args...>::value,
                    "Type T must be constructible with the given arguments.");

      auto managedArguments = getManagedArguments(std::forward<Args>(args)...);

      auto callback = [=] (Action action, ExecutionSpace space, void*& pointer) mutable {
         switch (action) {
            case ACTION_ALLOC:
            {
               switch (space) {
                  case CPU:
                     pointer = static_cast<void*>(new T(args...));
                     break;
                  case GPU:
                     pointer = static_cast<void*>(detail::make_on_device<T>(args...));
                     break;
               }

               break;
            }
            case ACTION_MOVE:
            {
               auto temp = managedArguments;
               (void)temp;
               break;
            }
            case ACTION_FREE:
            {
               switch (space) {
                  case CPU:
                  {
                     delete static_cast<T*>(pointer);
                     break;
                  }
                  case GPU:
                  {
                     T* typedPointer = static_cast<T*>(pointer);
                     detail::destroy_on_device<<<1, 1>>>(typedPointer);
                     break;
                  }
                  case NONE:
                  {
                     freeManagedArrays(managedArguments);
                     break;
                  }
               }

               break;
            }
         }
      };

      return managed_ptr<T>(callback);
   }

   template <typename T>
   CHAI_HOST_DEVICE T getRawPointers(T arg) {
      return arg;
   }

   template <typename T>
   CHAI_HOST_DEVICE T* getRawPointers(ManagedArray<T> arg) {
      return (T*) arg;
   }

   template <typename T>
   CHAI_HOST_DEVICE T* getRawPointers(managed_ptr<T> arg) {
      return arg.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @param[in] args The arguments to T's constructor
   ///
   template <typename T,
             typename... Args,
             typename std::enable_if<!std::is_constructible<T, Args...>::value, int>::type = 0>
   CHAI_HOST managed_ptr<T> make_managed(Args&&... args) {
      auto managedArguments = getManagedArguments(args...);

      auto callback = [=] (Action action, ExecutionSpace space, void*& pointer) mutable {
         switch (action) {
            case ACTION_ALLOC:
            {
               switch (space) {
                  case CPU:
                     pointer = static_cast<void*>(new T(getRawPointers(args)...));
                     break;
                  case GPU:
                     pointer = static_cast<void*>(detail::make_on_device<T>(getRawPointers(args)...));
                     break;
               }

               break;
            }
            case ACTION_MOVE:
            {
               auto temp = managedArguments;
               (void)temp;
               break;
            }
            case ACTION_FREE:
            {
               switch (space) {
                  case CPU:
                  {
                     delete static_cast<T*>(pointer);
                     break;
                  }
                  case GPU:
                  {
                     T* typedPointer = static_cast<T*>(pointer);
                     detail::destroy_on_device<<<1, 1>>>(typedPointer);
                     break;
                  }
                  case NONE:
                  {
                     freeManagedArrays(managedArguments);
                     break;
                  }
               }

               break;
            }
         }
      };

      return managed_ptr<T>(callback);
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @param[in] f The factory function that will create the object
   /// @param[in] args The arguments to the factory function
   ///
   template <typename T,
             typename F,
             typename... Args>
   CHAI_HOST managed_ptr<T> make_managed_from_factory(F&& f, Args&&... args) {
      static_assert(std::is_pointer<typename std::result_of<F(Args...)>::type>::value,
                    "Factory function must return a pointer");

      using R = typename std::remove_pointer<typename std::result_of<F(Args...)>::type>::type;

      static_assert(std::is_convertible<R*, T*>::value,
                    "Factory function must return a type that is convertible to T*.");

      auto managedArguments = getManagedArguments(args...);

      auto callback = [=] (Action action, ExecutionSpace space, void*& pointer) mutable {
         switch (action) {
            case ACTION_ALLOC:
            {
               switch (space) {
                  case CPU:
                     pointer = static_cast<void*>(f(args...));
                     break;
                  case GPU:
                     pointer = static_cast<void*>(detail::make_on_device_from_factory<R>(f, args...));
                     break;
               }

               break;
            }
            case ACTION_MOVE:
            {
               auto temp = managedArguments;
               (void)temp;
               break;
            }
            case ACTION_FREE:
            {
               switch (space) {
                  case CPU:
                  {
                     delete static_cast<T*>(pointer);
                     break;
                  }
                  case GPU:
                  {
                     T* typedPointer = static_cast<T*>(pointer);
                     detail::destroy_on_device<<<1, 1>>>(typedPointer);
                     break;
                  }
                  case NONE:
                  {
                     freeManagedArrays(managedArguments);
                     break;
                  }
               }

               break;
            }
         }
      };

      return managed_ptr<T>(callback);
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using static_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using static_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> static_pointer_cast(const managed_ptr<U>& other) noexcept {
      auto callback = [=] (Action action, ExecutionSpace space, void*& pointer) mutable {
         if (other.m_pointer_record) {
            other.m_pointer_record->m_callback(action, space, pointer);

            switch (action) {
               case ACTION_ALLOC:
               {
                  other.m_pointer_record->m_callback(action, space, pointer);
                  U* oldPointer = static_cast<U*>(pointer);
                  T* newPointer;

                  switch (space) {
                     case CPU:
                        newPointer = static_cast<T*>(oldPointer);
                        break;
                     case GPU:
                        newPointer = detail::static_pointer_cast_on_device<T>(other);
                        break;
                  }

                  break;
               }
               default:
                  break;
            }
         }
      };

      auto hostPtr = static_cast<T*>(other.get());
      auto devicePtr = detail::static_pointer_cast_on_device<T>(other);
      return managed_ptr<T>(other, {hostPtr, devicePtr});
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using dynamic_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using dynamic_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> dynamic_pointer_cast(const managed_ptr<U>& other) noexcept {
      static_assert(true, "CUDA does not support dynamic_cast");
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using const_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using const_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> const_pointer_cast(const managed_ptr<U>& other) noexcept {
      auto hostPtr = const_cast<T*>(other.get());
      auto devicePtr = detail::const_pointer_cast_on_device<T>(other);
      return managed_ptr<T>(other, {hostPtr, devicePtr});
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using reinterpret_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using reinterpret_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> reinterpret_pointer_cast(const managed_ptr<U>& other) noexcept {
      auto hostPtr = reinterpret_cast<T*>(other.get());
      auto devicePtr = detail::reinterpret_pointer_cast_on_device<T>(other);
      return managed_ptr<T>(other, {hostPtr, devicePtr});
   }
   
#endif // __CUDACC__

   /// Comparison operators

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to compare
   /// @param[in] rhs The second managed_ptr to compare
   ///
   template <class T, class U>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(const managed_ptr<T>& lhs, const managed_ptr<U>& rhs) noexcept {
      return lhs.get() == rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to compare
   /// @param[in] rhs The second managed_ptr to compare
   ///
   template <class T, class U>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(const managed_ptr<T>& lhs, const managed_ptr<U>& rhs) noexcept {
      return lhs.get() != rhs.get();
   }

   /// Comparison operators with nullptr

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] lhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(const managed_ptr<T>& lhs, std::nullptr_t) noexcept {
      return lhs.get() == nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] rhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
      return nullptr == rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison with nullptr.
   ///
   /// @param[in] lhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(const managed_ptr<T>& lhs, std::nullptr_t) noexcept {
      return lhs.get() != nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison with nullptr.
   ///
   /// @param[in] rhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
      return nullptr != rhs.get();
   }

   template <class T>
   void swap(managed_ptr<T>& lhs, managed_ptr<T>& rhs) noexcept {
      std::swap(lhs.m_active_pointer, rhs.m_active_pointer);
      std::swap(lhs.m_pointer_record, rhs.m_pointer_record);
   }

} // namespace chai

#endif // MANAGED_PTR

